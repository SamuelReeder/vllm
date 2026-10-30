# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel's copies of Qwen3.8's MXFP4 linear weights, packed from the loaded model.

Per layer the kernel streams four GEMVs (N x K at TP 1):
  GDN layer:  inp  = [in_proj_qkvz | in_proj_ba]      (16480 x 5120)
              out  = out_proj                         (5120 x 6144)
  attn layer: qkv  = qkv_proj                         (14336 x 5120)
              o    = o_proj                           (5120 x 6144)
  every layer gu   = gate_up_proj, rows interleaved per 8 columns
              down = down_proj                        (5120 x 17408)
Each is re-tiled for the gfx942 FP8 MFMA (mkcore/fp4.py) and stored in the order the
waves stream it. vLLM keeps its own tensors for the prefill path, so the copies add the
linear weights once more (12.9 GB at TP 1, half of it per rank at TP 2). The embedding
table, the final norm and the small per-layer tensors are read in place.

At TP 2 each rank packs the shards vLLM loaded on it. vLLM's TP layout is the kernel's
per-rank layout (kernels/dims.py ``Dims``): column-parallel layers hold the rank's
slice of each output segment in segment order (in_proj_qkvz: q | k | v | z, in_proj_ba:
b | a, qkv_proj: q and gate | k | v, gate_up_proj: gate | up), row-parallel layers
(out_proj, o_proj, down_proj) the rank's K slice of the packed bytes and scales.
"""

import torch

from ..kernels.dims import DV, HD, NL, Dims, H, is_attn
from ..kernels.kernel import chunks
from ..kernels.mkcore import fp4
from ..kernels.mkcore.pack import gu_order, unit_order
from ..spec import GEMVS, mxfp4_parts

SMALL = {
    "gdn": (
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.conv1d.weight",
        "linear_attn.norm.weight",
    ),
    "attn": ("self_attn.q_norm.weight", "self_attn.k_norm.weight"),
}


def gemv_shapes(d: Dims) -> dict[str, tuple[int, int]]:
    """(N, K) of each GEMV on one rank."""
    return {
        "inp": (d.inp_n, H),
        "out": (H, d.val_dim),
        "qkv": (d.qkv_n, H),
        "o": (H, d.nq * HD),
        "gu": (2 * d.ffn, H),
        "down": (H, d.ffn),
    }


def small_rows(d: Dims) -> dict[str, int]:
    """Rows of each small per-layer tensor on one rank (norm weights: replicated)."""
    return {
        "linear_attn.A_log": d.nvh,
        "linear_attn.dt_bias": d.nvh,
        "linear_attn.conv1d.weight": d.conv_dim,
        "linear_attn.norm.weight": DV,  # gated RMSNorm over one value head
        "self_attn.q_norm.weight": HD,
        "self_attn.k_norm.weight": HD,
        "input_layernorm.weight": H,
        "post_attention_layernorm.weight": H,
    }


class PackedWeights:
    """The kernel's weights of one TP rank: packed GEMV tiles per (layer, GEMV) and the
    in-place small tensors per (layer, name)."""

    def __init__(self, layers, tp: int = 1):
        d = Dims(tp)
        shapes, rows = gemv_shapes(d), small_rows(d)
        self.chunks: tuple = ()  # Cfg.chunks: no override of kernel.chunks(tp)
        C_of = chunks(tp, self.chunks)
        self.mats: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.small: dict[tuple[int, str], torch.Tensor] = {}
        for i in range(NL):
            layer = layers[i]
            kind = "attn" if is_attn(i) else "gdn"
            for group in (kind, "mlp"):
                for name, parts in GEMVS[group].items():
                    ws = [mxfp4_parts(layer.get_submodule(p)) for p in parts]
                    assert all(x is not None for x in ws), (i, name)
                    w = torch.cat([x[0] for x in ws])
                    s = torch.cat([x[1] for x in ws])
                    N, K = shapes[name]
                    # a replicated (not TP-sharded) part would give a wider N
                    assert w.shape == (N, K // 2), (i, name, tuple(w.shape), tp)
                    if name == "gu":
                        w, s = gu_order(w), gu_order(s)
                    tiles, scales = fp4.repack(w, s)
                    C = C_of.get(name, 1)
                    self.mats[(i, name)] = (
                        unit_order(tiles, 2, C),
                        unit_order(scales, 1, C),
                    )
            names = (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ) + SMALL[kind]
            for n in names:
                mod, _, attr = n.rpartition(".")
                t = getattr(layer.get_submodule(mod), attr)
                assert t.shape[0] == rows[n], (i, n, tuple(t.shape), tp)
                # the kernel reads the small tensors as bf16 (the checkpoint's dtype)
                self.small[(i, n)] = t.detach().to(torch.bfloat16).contiguous()
