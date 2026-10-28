# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel's copies of Gemma 4 26B-A4B's linear weights, packed from the model.

Per layer the kernel streams (kernels/pack.py):
  qkv    = qkv_proj (sliding: [q | k | v]; full: its [q | k] rows)
                                              fp8  (8192 / 9216 x 2816)
  o      = o_proj                             fp8  (2816 x 4096 / 8192)
  router = router.proj                        bf16 (128 x 2816)
  e1     = experts' w13, then the dense MLP's gate_up_proj as 3 pseudo-experts of 704
           columns, [gate; up] rows in gu_order  fp8 (131 x 1408 x 2816)
  e2     = the same experts' down projections, K padded 704 -> 768 with zeros
                                              fp8  (131 x 2816 x 768)
Each is re-tiled for the gfx942 MFMA and stored in the order the waves stream it. vLLM
keeps its own tensors for the prefill path, so the copies add the linear weights once
more (about 25 GB). The embedding table and the final norm are read in place (bf16);
the small per-layer tensors are copied as bf16.

vLLM keeps the FP8 weights on gfx942 as e4m3fnuz: the checkpoint's e4m3fn bits with
0x80 mapped to 0 and the per-row scales doubled (normalize_e4m3fn_to_e4m3fnuz). The
kernel's packer starts from the checkpoint's e4m3fn bits and does the same mapping
itself, so it gets vLLM's bits as they are and vLLM's scales halved back; both steps
are exact.
"""

import torch

from ..kernels import pack
from ..kernels.dims import EFF, EFFP, FFN, NDS, NL, QKV_F, H, is_full
from ..spec import FP8_LINEARS, expert_parts, fp8_parts

SMALL = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "pre_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm_1.weight",
    "post_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm.weight",
    "router.scale",
    "router.per_expert_scale",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "layer_scalar",
)


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


def _srow(s: torch.Tensor) -> torch.Tensor:
    """Undo vLLM's doubling: per-row scales [.., N, 1] -> the checkpoint's [.., N]."""
    return s[..., 0] * 0.5


class PackedWeights:
    """The kernel's weights: packed GEMV (tiles, scales) per (layer, GEMV) (bf16
    GEMVs: (tiles, tiles)), the small tensors per (layer, name), and the bf16 ones
    the pointer table's row 0 ("layer -1") reads as its norms and layer scalar."""

    def __init__(self, layers, chunks: dict[str, int]):
        self.chunks = dict(chunks)
        self.mats: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.small: dict[tuple[int, str], torch.Tensor] = {}
        device = layers[0].input_layernorm.weight.device
        self.ones = torch.ones(H, dtype=torch.bfloat16, device=device)
        for i in range(NL):
            layer = layers[i]
            full = is_full(i)

            def lin(name: str, i=i, layer=layer, full=full):
                parts = fp8_parts(layer.get_submodule(name), FP8_LINEARS[name][full])
                assert parts is not None, (i, name)
                return _u8(parts[0]), _srow(parts[1])

            w, s = lin("self_attn.qkv_proj")
            if full:  # [q | k | v = k]: the kernel computes V from k
                w, s = w[:QKV_F], s[:QKV_F]
            self._fp8(i, "qkv", w, s)
            self._fp8(i, "o", *lin("self_attn.o_proj"))
            self._bf16(i, "router", layer.get_submodule("router.proj").weight)
            parts = expert_parts(layer.experts)
            assert parts is not None, i
            w13, s13, w2, s2 = parts
            # the dense MLP as NDS pseudo-experts of EFF columns after the experts
            gu, sgu = lin("mlp.gate_up_proj")
            dn, sdn = lin("mlp.down_proj")
            cols = [slice(k * EFF, (k + 1) * EFF) for k in range(NDS)]
            w = torch.cat(
                [_u8(w13)] + [torch.cat([gu[c], gu[FFN:][c]])[None] for c in cols]
            )
            s = torch.cat(
                [_srow(s13)] + [torch.cat([sgu[c], sgu[FFN:][c]])[None] for c in cols]
            )
            self._fp8(i, "e1", pack.gu_order(w), pack.gu_order_rows(s))
            w = torch.zeros(w.shape[0], H, EFFP, dtype=torch.uint8, device=w.device)
            w[: w2.shape[0], :, :EFF] = _u8(w2)
            for k, c in enumerate(cols):
                w[w2.shape[0] + k, :, :EFF] = dn[:, c]
            s = torch.cat([_srow(s2)] + [sdn[None]] * NDS)
            self._fp8(i, "e2", w, s)
            del w, s
            for n in SMALL:
                mod, _, attr = n.rpartition(".")
                t = getattr(layer.get_submodule(mod), attr)
                t = t.detach().to(torch.bfloat16).reshape(-1)
                if n == "layer_scalar":  # the kernel reads it as one dword
                    t = torch.cat([t, t])
                self.small[(i, n)] = t.contiguous()

    def _fp8(self, i: int, name: str, w: torch.Tensor, srow: torch.Tensor) -> None:
        tiles, scales = pack.fp8_tiles(w, srow)
        C = self.chunks[name]
        self.mats[(i, name)] = (
            pack.unit_order(tiles, 3, C),
            pack.unit_order(scales, 1, C),
        )

    def _bf16(self, i: int, name: str, w: torch.Tensor) -> None:
        t = pack.unit_order(pack.pack_bf16(w), 3, self.chunks[name])
        self.mats[(i, name)] = (t, t)
