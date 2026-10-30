# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 26B-A4B's whole-model decode step on gfx942: one persistent launch per step.

The kernel (kernels/kernel.py) runs the embedding, all 30 layers (attention, the dense
MLP, router and top-8 routed experts) and the final norm, and returns the normed hidden
states; vLLM's LM head (with the logit soft-cap) and sampler follow. It reads and
writes vLLM's own paged KV caches.

Everything a launch reads sits at fixed addresses: ``run`` copies the step's tokens,
positions, slot mappings and block tables into the kernel's own buffers (the copies
are captured with the launch in a FULL CUDA graph), and the grid barrier's epoch
lives on the device. The sliding-window and full attention layers have their own KV
cache groups (their pages share the cache memory, so a request has its own blocks in
each group): every layer reads the block table and slot mapping of its group. A row
vLLM pads a step with (slot mapping -1) writes no KV and selects no routed expert.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.common.specialized_decode.spec import Step

from ..kernels import kernel as KK
from ..kernels import moe as MO
from ..kernels.dims import (
    EFFP,
    HD_F,
    HD_S,
    NKV_F,
    NKV_S,
    NL,
    NQ,
    QKV_F,
    ROT_F,
    THETA_F,
    THETA_S,
    H,
    is_full,
)
from ..spec import attention_backend, kv_geometry
from .weights import PackedWeights

logger = init_logger(__name__)


def rope_table(full: bool, maxlen: int, device) -> torch.Tensor:
    """[maxlen][HD] fp32: cos of the HD / 2 angle pairs (NeoX pairs d, d + HD / 2),
    then their sin. Full layers' proportional RoPE rotates the first ROT_F pairs; the
    rest have angle 0 (cos 1, sin 0)."""
    hd, theta = (HD_F, THETA_F) if full else (HD_S, THETA_S)
    i = torch.arange(hd // 2, dtype=torch.float64, device=device)
    inv = theta ** (-2 * i / hd)
    if full:
        inv = torch.where(i < ROT_F, inv, torch.zeros_like(inv))
    ang = torch.arange(maxlen, dtype=torch.float64, device=device)[:, None] * inv[None]
    return torch.cat([ang.cos().float(), ang.sin().float()], 1).contiguous()


def default_chunks() -> dict[str, int]:
    """The GEMV K chunks of every Cfg built here (the packed weights depend on them)."""
    return {
        f.name[2:]: f.default
        for f in dataclasses.fields(KK.Cfg)
        if f.name.startswith("c_")
    }


class Gemma4Gfx942Decode:
    """The kernel of every served width, the buffers they share, and the cache
    bindings."""

    def __init__(self, vllm_config, lm, widths: tuple[int, ...]):
        """lm: the loaded Gemma4ForCausalLM (``lm.model`` is the backbone)."""
        self.vllm_config = vllm_config
        self.widths = widths
        self.model = lm.model
        device = self.model.embed_tokens.weight.device
        self.device = device
        layers = self.model.layers
        self.attn_names = [layers[i].self_attn.attn.layer_name for i in range(NL)]
        self.cache_layers = tuple(self.attn_names)
        self.maxlen = vllm_config.model_config.max_model_len
        self.cus = torch.cuda.get_device_properties(device).multi_processor_count
        self.chunks = default_chunks()
        self.weights = PackedWeights(layers, self.chunks)
        Mx = max(widths)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.buf = {k: z(Mx, H) for k in ("h", "h1", "a", "d", "e")}
        self.ssq = {k: z(Mx, H // 16) for k in ("a", "d", "e")}
        self.rstd1 = z(Mx)
        self.y = z(Mx, QKV_F)
        self.o = z(Mx, NQ * HD_F)
        self.router = z(1, Mx, MO.LGN)
        self.act = z(Mx, MO.NSLOT, EFFP)  # columns EFF.. stay 0 (the padded down K)
        self.rowv = z(Mx, dt=torch.int32)
        # router_publish: the gate_up inputs, fp8 [2M][H] and scales [2M][H / 128];
        # combine_rows: the next layer's qkv input, fp8 [M][H] and scales [M][H / 128]
        self.xg = z(2 * Mx * H // 4, dt=torch.int32)
        self.xgs = z(2 * Mx * (H // 128))
        self.xq = z(Mx * H // 4, dt=torch.int32)
        self.xqs = z(Mx * (H // 128))
        self.cnt = z(len(KK.CNT_REGION) * KK.CNT_STRIDE, dt=torch.int32)
        self.flags = z((self.cus + KK.EPOCH_WORD_LINE + 1) * 32, dt=torch.int32)
        self.tok = z(Mx, dt=torch.int32)
        self.pos = z(Mx, dt=torch.int32)
        self.out = z(Mx, H, dt=self.model.embed_tokens.weight.dtype)
        self.rope_s = rope_table(False, self.maxlen, device)
        self.rope_f = rope_table(True, self.maxlen, device)
        self.launches: dict[int, Any] = {}
        self.cfgs: dict[int, KK.Cfg] = {}
        self.part: torch.Tensor | None = None
        self.rec: torch.Tensor | None = None
        # the KV cache groups (layer names), from the model runner before binding
        self.cache_groups: list[list[str]] = []
        self.attn_groups: list[str] = []  # each group's first layer: its metadata key
        # no state-space layers (the shared skip_step reads their groups)
        self.gdn_groups: list[str] = []
        self.bt: list[torch.Tensor] = []  # per group [Mx][block-table columns]
        self.slot: torch.Tensor | None = None  # [groups][Mx]
        self.tab: torch.Tensor | None = None
        self.glob: torch.Tensor | None = None

    # ---------------------------------------------------------------- caches
    def set_cache_groups(self, groups: list[list[str]]) -> None:
        """The layer names of each KV cache group of the model runner."""
        self.cache_groups = [list(g) for g in groups]

    def _group_of(self, names: list[str]) -> tuple[list[str], list[int]]:
        """(first layer of each group holding some of `names`, each name's group)."""
        of = {n: gi for gi, g in enumerate(self.cache_groups) for n in g}
        missing = [n for n in names if n not in of]
        assert not missing, f"layers in no KV cache group: {missing[:4]}"
        firsts: list[str] = []
        index: dict[int, int] = {}
        for n in names:
            g = of[n]
            if g not in index:
                index[g] = len(firsts)
                firsts.append(n)
        return firsts, [index[of[n]] for n in names]

    def bind_caches(self, caches: Mapping[str, Any]) -> None:
        sfc = self.vllm_config.compilation_config.static_forward_context
        geo: dict[bool, tuple[int, ...]] = {}
        for full, nkv, hd in ((False, NKV_S, HD_S), (True, NKV_F, HD_F)):
            name = self.attn_names[5 if full else 0]
            backend = attention_backend(sfc[name])
            geometry = kv_geometry(caches[name], backend, nkv, hd)
            assert geometry is not None and not geometry[5], (name, backend)
            block, s0, s1, s2, sv, _ = geometry
            geo[full] = (block, -(-self.maxlen // block), s0, s1, s2, sv)
        (bs, mbs, *kv_s), (bf, mbf, *kv_f) = geo[False], geo[True]
        for w in self.widths:
            cfg = KK.Cfg(
                w,
                self.cus,
                self.maxlen,
                bs,
                mbs,
                tuple(kv_s),
                bf,
                mbf,
                tuple(kv_f),
                lm_head=False,
            )
            assert cfg.chunks() == self.chunks, (cfg.chunks(), self.chunks)
            if self.cfgs.get(w) != cfg:
                self.cfgs[w] = cfg
                self.launches[w] = KK.build(cfg)
        Mx = max(self.widths)
        cfg = self.cfgs[Mx]
        self.part = torch.zeros(
            max(c.part_floats() for c in self.cfgs.values()), device=self.device
        )
        self.rec = torch.zeros(
            max(cfg.geo(False).nrec(Mx), cfg.geo(True).nrec(Mx)), device=self.device
        )
        self.attn_groups, group = self._group_of(self.attn_names)
        full_of = [None] * len(self.attn_groups)
        for i, g in enumerate(group):
            assert full_of[g] in (None, is_full(i)), "a cache group mixes layer types"
            full_of[g] = is_full(i)
        zi = lambda *s: torch.zeros(*s, dtype=torch.int32, device=self.device)  # noqa: E731
        self.bt = [zi(Mx, mbf if f else mbs) for f in full_of]
        self.slot = zi(len(self.attn_groups), Mx)
        m, W = self.model, self.weights
        # G_LM and G_SPARE: the LM head and debug_route only
        g = [0] * (KK.NGLOB + KK.NGEO)
        # the run-time cache geometry (Cfg.rt_geom): the same for every width
        g[KK.G_GEOM :] = cfg.geom_slots()
        g[KK.G_EMBED] = m.embed_tokens.weight.data_ptr()
        g[KK.G_FNORM] = m.norm.weight.data_ptr()
        g[KK.G_ROPE_S] = self.rope_s.data_ptr()
        g[KK.G_ROPE_F] = self.rope_f.data_ptr()
        for k, slot in (
            ("h", KK.G_H),
            ("h1", KK.G_H1),
            ("a", KK.G_A),
            ("d", KK.G_D),
            ("e", KK.G_E),
        ):
            g[slot] = self.buf[k].data_ptr()
        for k, slot in (("a", KK.G_SSQA), ("d", KK.G_SSQD), ("e", KK.G_SSQE)):
            g[slot] = self.ssq[k].data_ptr()
        g[KK.G_RSTD1] = self.rstd1.data_ptr()
        g[KK.G_Y] = self.y.data_ptr()
        g[KK.G_O] = self.o.data_ptr()
        g[KK.G_PART] = self.part.data_ptr()
        g[KK.G_CNT] = self.cnt.data_ptr()
        g[KK.G_FLAGS] = self.flags.data_ptr()
        g[KK.G_REC] = self.rec.data_ptr()
        g[KK.G_TOK] = self.tok.data_ptr()
        g[KK.G_POS] = self.pos.data_ptr()
        g[KK.G_OUT] = self.out.data_ptr()
        g[KK.G_LOGIT] = self.router.data_ptr()
        g[KK.G_ACT] = self.act.data_ptr()
        g[KK.G_ROWV] = self.rowv.data_ptr()
        g[KK.G_XG] = self.xg.data_ptr()
        g[KK.G_XGS] = self.xgs.data_ptr()
        g[KK.G_XQ] = self.xq.data_ptr()
        g[KK.G_XQS] = self.xqs.data_ptr()
        self.glob = torch.tensor(g, dtype=torch.int64, device=self.device)
        tab = torch.zeros(NL + 1, KK.NTAB, dtype=torch.int64)
        # row 0, "layer -1": ones as its post-feedforward norms, layer scalar 1.0
        for k in (KK.T_POST1, KK.T_POST2, KK.T_POSTF, KK.T_LS):
            tab[0, k] = W.ones.data_ptr()
        for i in range(NL):
            sm = lambda n: W.small[(i, n)].data_ptr()  # noqa: E731,B023
            mp = lambda n: [t.data_ptr() for t in W.mats[(i, n)]]  # noqa: E731,B023
            r = tab[i + 1]
            r[KK.T_QKV_T], r[KK.T_QKV_S] = mp("qkv")
            r[KK.T_O_T], r[KK.T_O_S] = mp("o")
            r[KK.T_RT] = mp("router")[0]
            r[KK.T_E1_T], r[KK.T_E1_S] = mp("e1")
            r[KK.T_E2_T], r[KK.T_E2_S] = mp("e2")
            r[KK.T_IN_LN] = sm("input_layernorm.weight")
            r[KK.T_PA_LN] = sm("post_attention_layernorm.weight")
            r[KK.T_PF_LN] = sm("pre_feedforward_layernorm.weight")
            r[KK.T_PF2_LN] = sm("pre_feedforward_layernorm_2.weight")
            r[KK.T_RSCALE] = sm("router.scale")
            r[KK.T_PES] = sm("router.per_expert_scale")
            r[KK.T_QN] = sm("self_attn.q_norm.weight")
            r[KK.T_KN] = sm("self_attn.k_norm.weight")
            r[KK.T_POST1] = sm("post_feedforward_layernorm_1.weight")
            r[KK.T_POST2] = sm("post_feedforward_layernorm_2.weight")
            r[KK.T_POSTF] = sm("post_feedforward_layernorm.weight")
            r[KK.T_LS] = sm("layer_scalar")
            r[KK.T_KV] = caches[self.attn_names[i]].data_ptr()
            r[KK.T_BT] = self.bt[group[i]].data_ptr()
            r[KK.T_SLOT] = self.slot[group[i]].data_ptr()
        self.tab = tab.to(self.device)
        logger.info(
            "Gemma 4 26B-A4B gfx942 decode: KV blocks %d (sliding) / %d (full), "
            "%d KV cache groups, widths %s",
            bs,
            bf,
            len(self.attn_groups),
            list(self.widths),
        )

    def steps_run(self) -> int:
        """Decode steps the kernel has run since the caches were bound: the grid
        barrier's epoch on the device advances by one step's barriers per launch."""
        word = (self.cus + KK.EPOCH_WORD_LINE) * 32
        return int(self.flags[word].item()) // KK.barriers_per_step(NL)

    # ---------------------------------------------------------------- step
    def stage(self, step: Step, input_ids, inputs_embeds) -> torch.Tensor:
        """Copy the step's inputs and cache indices into the kernel's buffers; returns
        the global table the launch reads."""
        W = step.kernel_width
        assert self.slot is not None and self.glob is not None
        # spec._refuse_config: the runner hands this model token ids, not embeddings
        assert inputs_embeds is None and input_ids is not None
        md = get_forward_context().attn_metadata
        self.pos[:W].copy_(step.positions[:W])
        for g, name in enumerate(self.attn_groups):
            amd = md[name]
            bt = self.bt[g]
            nb = min(amd.block_table.shape[1], bt.shape[1])
            self.slot[g, :W].copy_(amd.slot_mapping[:W])
            bt[:W, :nb].copy_(amd.block_table[:W, :nb])
        # a padding row (slot mapping -1) selects no routed expert
        self.rowv[:W].copy_(self.slot[0, :W] >= 0)
        self.tok[:W].copy_(input_ids[:W])
        return self.glob

    def launch(self, width: int, glob: torch.Tensor, layers: int = NL) -> torch.Tensor:
        assert self.tab is not None
        self.launches[width](
            self.tab.data_ptr(),
            glob.data_ptr(),
            layers,
            stream=torch.cuda.current_stream(),
        )
        return self.out[:width]

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> torch.Tensor:
        glob = self.stage(step, input_ids, inputs_embeds)
        return self.launch(step.kernel_width, glob)
