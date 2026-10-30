# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.6-35B-A3B's whole-model decode step on gfx942: one persistent launch per step.

The kernel (kernels/kernel.py) runs the embedding, all 40 layers (mixer, router, top-8
routed experts and the shared expert) and the final norm, and returns the normed hidden
states; vLLM's LM head and sampler follow. It reads and writes vLLM's own caches: the
paged KV of the 10 attention layers and the conv / SSM state slots of the 30 Gated
DeltaNet layers.

Everything a launch reads sits at fixed addresses: ``run`` copies the step's tokens,
positions, slot mappings, block tables and state indices into the kernel's own buffers
(the copies are captured with the launch in a FULL CUDA graph), and the grid
barrier's epoch lives on the device. A hybrid model's KV cache groups share the cache
memory, so a request has its own blocks and state slots in each group: every layer
reads the indices of its group. A row vLLM pads a step with (slot mapping -1) writes no
KV and selects no routed expert.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.common.specialized_decode.spec import Step

from ..kernels import kernel as KK
from ..kernels import mixer as X
from ..kernels import moe as MO
from ..kernels.dims import (
    FI,
    INP_N,
    NKV,
    NL,
    NVH,
    QKV_N,
    ROT,
    THETA,
    VAL_DIM,
    H,
    is_attn,
)
from ..spec import HEAD_DIM, attention_backend, kv_geometry
from .weights import PackedWeights

logger = init_logger(__name__)


def rope_table(maxlen: int, device) -> torch.Tensor:
    """[maxlen][64] fp32: cos then sin of the 32 rotary frequencies (NeoX RoPE on the
    first ROT dims of a head)."""
    inv = THETA ** (-torch.arange(0, ROT, 2, dtype=torch.float64, device=device) / ROT)
    ang = torch.arange(maxlen, dtype=torch.float64, device=device)[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], 1).float().contiguous()


def default_chunks() -> dict[str, int]:
    """The GEMV K chunks of every Cfg built here (the packed weights depend on them)."""
    return {
        f.name[2:]: f.default
        for f in dataclasses.fields(KK.Cfg)
        if f.name.startswith("c_")
    }


class Qwen35MoeGfx942Decode:
    """The kernel of every served width, the buffers they share, and the cache
    bindings."""

    def __init__(self, vllm_config, lm, widths: tuple[int, ...]):
        """lm: the loaded Qwen3_5MoeForCausalLM (``lm.model`` is the backbone)."""
        self.vllm_config = vllm_config
        self.widths = widths
        self.model = lm.model
        device = self.model.embed_tokens.weight.device
        self.device = device
        layers = self.model.layers
        self.attn_names = [
            layers[i].self_attn.attn.layer_name for i in range(NL) if is_attn(i)
        ]
        self.gdn_names = [
            layers[i].linear_attn.prefix for i in range(NL) if not is_attn(i)
        ]
        self.cache_layers = tuple(self.attn_names + self.gdn_names)
        self.maxlen = vllm_config.model_config.max_model_len
        self.cus = torch.cuda.get_device_properties(device).multi_processor_count
        self.chunks = default_chunks()
        self.weights = PackedWeights(layers, self.chunks)
        Mx = max(widths)
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.hb = z(3, Mx, H)
        self.ssq = z(2, Mx, H // 16)
        self.y = z(Mx, max(INP_N, QKV_N))
        self.y2 = z(Mx, VAL_DIM)
        self.hssq = z(Mx, NVH, 8)
        self.router = z(Mx, MO.LGN)
        self.act = z(Mx, MO.NSLOT, FI)
        self.rowv = z(Mx, dt=torch.int32)
        self.cnt = z(len(KK.CNT_REGION) * KK.CNT_STRIDE, dt=torch.int32)
        self.flags = z((self.cus + KK.EPOCH_WORD_LINE + 1) * 32, dt=torch.int32)
        self.tok = z(Mx, dt=torch.int32)
        self.pos = z(Mx, dt=torch.int32)
        self.out = z(Mx, H, dt=self.model.embed_tokens.weight.dtype)
        # a step handed embeddings (a multimodal model's decode) reads them as a table
        # of Mx rows indexed by row
        self.emb = z(Mx, H, dt=self.model.embed_tokens.weight.dtype)
        self.iota = torch.arange(Mx, dtype=torch.int32, device=device)
        self.rope = rope_table(self.maxlen, device)
        self.launches: dict[int, Any] = {}
        self.cfgs: dict[int, KK.Cfg] = {}
        self.part: torch.Tensor | None = None
        self.rec: torch.Tensor | None = None
        # the KV cache groups (layer names), from the model runner before binding
        self.cache_groups: list[list[str]] = []
        self.attn_groups: list[str] = []  # each group's first layer: its metadata key
        self.gdn_groups: list[str] = []
        self.bt: torch.Tensor | None = None
        self.slot: torch.Tensor | None = None
        self.sidx: torch.Tensor | None = None
        self.tab: torch.Tensor | None = None
        self.glob: torch.Tensor | None = None
        self.glob_emb: torch.Tensor | None = None

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
        kv0 = caches[self.attn_names[0]]
        conv0, ssm0 = caches[self.gdn_names[0]]
        sfc = self.vllm_config.compilation_config.static_forward_context
        backend = attention_backend(sfc[self.attn_names[0]])
        geometry = kv_geometry(kv0, backend, NKV, HEAD_DIM)
        assert geometry is not None
        block, s0, s1, s2, sv, xpack = geometry
        maxblocks = -(-self.maxlen // block)
        for w in self.widths:
            cfg = KK.Cfg(
                w,
                self.cus,
                self.maxlen,
                block,
                maxblocks,
                s0,
                s1,
                s2,
                sv,
                xpack,
                conv0.stride(0) * conv0.element_size(),
                ssm0.stride(0) * ssm0.element_size(),
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
        self.rec = torch.zeros(Mx * NKV * cfg.ncmax * X.REC, device=self.device)
        self.attn_groups, attn_group = self._group_of(self.attn_names)
        self.gdn_groups, gdn_group = self._group_of(self.gdn_names)
        zi = lambda *s: torch.zeros(*s, dtype=torch.int32, device=self.device)  # noqa: E731
        self.bt = zi(len(self.attn_groups), Mx, maxblocks)
        self.slot = zi(len(self.attn_groups), Mx)
        self.sidx = zi(len(self.gdn_groups), Mx)
        m, W = self.model, self.weights
        g = [0] * (KK.NGLOB + KK.NGEO)
        # the run-time cache geometry (Cfg.rt_geom): the same for every width
        g[KK.G_GEOM :] = cfg.geom_slots()
        g[KK.G_EMBED] = m.embed_tokens.weight.data_ptr()
        g[KK.G_FNORM] = m.norm.weight.data_ptr()
        g[KK.G_ROPE] = self.rope.data_ptr()
        g[KK.G_HB] = self.hb.data_ptr()
        g[KK.G_SSQ] = self.ssq.data_ptr()
        g[KK.G_Y] = self.y.data_ptr()
        g[KK.G_Y2] = self.y2.data_ptr()
        g[KK.G_HSSQ] = self.hssq.data_ptr()
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
        self.glob = torch.tensor(g, dtype=torch.int64, device=self.device)
        g[KK.G_EMBED], g[KK.G_TOK] = self.emb.data_ptr(), self.iota.data_ptr()
        self.glob_emb = torch.tensor(g, dtype=torch.int64, device=self.device)
        tab = torch.zeros(NL, KK.NTAB, dtype=torch.int64)
        ai = gi = 0
        for i in range(NL):
            sm = lambda n: W.small[(i, n)].data_ptr()  # noqa: E731,B023
            mp = lambda n: [t.data_ptr() for t in W.mats[(i, n)]]  # noqa: E731,B023
            if is_attn(i):
                mt, ms = mp("qkv")
                ot, os_ = mp("o")
                tab[i, KK.T_QN] = sm("self_attn.q_norm.weight")
                tab[i, KK.T_KN] = sm("self_attn.k_norm.weight")
                tab[i, KK.T_KV] = caches[self.attn_names[ai]].data_ptr()
                tab[i, KK.T_IDX] = self.bt[attn_group[ai]].data_ptr()
                tab[i, KK.T_SLOT] = self.slot[attn_group[ai]].data_ptr()
                ai += 1
            else:
                mt, ms = mp("inp")
                ot, os_ = mp("out")
                conv, ssm = caches[self.gdn_names[gi]]
                tab[i, KK.T_BA] = mp("ba")[0]
                tab[i, KK.T_ALOG] = sm("linear_attn.A_log")
                tab[i, KK.T_DTB] = sm("linear_attn.dt_bias")
                tab[i, KK.T_CONVW] = sm("linear_attn.conv1d.weight")
                tab[i, KK.T_NORMW] = sm("linear_attn.norm.weight")
                tab[i, KK.T_CONV], tab[i, KK.T_SSM] = conv.data_ptr(), ssm.data_ptr()
                tab[i, KK.T_IDX] = self.sidx[gdn_group[gi]].data_ptr()
                gi += 1
            tab[i, KK.T_MIX_T], tab[i, KK.T_MIX_S] = mt, ms
            tab[i, KK.T_OUT_T], tab[i, KK.T_OUT_S] = ot, os_
            tab[i, KK.T_RT] = mp("router")[0]
            tab[i, KK.T_E1_T], tab[i, KK.T_E1_S] = mp("e1")
            tab[i, KK.T_E2_T], tab[i, KK.T_E2_S] = mp("e2")
            tab[i, KK.T_IN_LN] = sm("input_layernorm.weight")
            tab[i, KK.T_POST_LN] = sm("post_attention_layernorm.weight")
        self.tab = tab.to(self.device)
        logger.info(
            "Qwen3.6-35B-A3B gfx942 decode: %s KV block %d, %d block-table columns, "
            "%d attention and %d GDN cache groups, widths %s",
            backend,
            block,
            maxblocks,
            len(self.attn_groups),
            len(self.gdn_groups),
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
        the global table the launch reads (token ids or the given embeddings)."""
        W = step.kernel_width
        assert self.bt is not None and self.slot is not None and self.sidx is not None
        assert self.glob is not None and self.glob_emb is not None
        md = get_forward_context().attn_metadata
        pos = step.positions
        if pos.dim() == 2:  # M-RoPE [3, N]: a text token has three equal rows
            pos = pos[0]
        self.pos[:W].copy_(pos[:W])
        for g, name in enumerate(self.attn_groups):
            amd = md[name]
            nb = min(amd.block_table.shape[1], self.bt.shape[2])
            self.slot[g, :W].copy_(amd.slot_mapping[:W])
            self.bt[g, :W, :nb].copy_(amd.block_table[:W, :nb])
        # a padding row (slot mapping -1) selects no routed expert
        self.rowv[:W].copy_(self.slot[0, :W] >= 0)
        for g, name in enumerate(self.gdn_groups):
            self.sidx[g, :W].copy_(md[name].non_spec_state_indices_tensor[:W])
        if inputs_embeds is None:
            self.tok[:W].copy_(input_ids[:W])
            return self.glob
        self.emb[:W].copy_(inputs_embeds[:W])
        return self.glob_emb

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
