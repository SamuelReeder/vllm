# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel's copies of Qwen3.8's MXFP4 linear weights, packed from the loaded model.

Per layer the kernel streams four GEMVs:
  GDN layer:  inp  = [in_proj_qkvz | in_proj_ba]      (16480 x 5120)
              out  = out_proj                         (5120 x 6144)
  attn layer: qkv  = qkv_proj                         (14336 x 5120)
              o    = o_proj                           (5120 x 6144)
  every layer gu   = gate_up_proj, rows interleaved per 8 columns
              down = down_proj                        (5120 x 17408)
Each is re-tiled for the gfx942 FP8 MFMA (kernels/fp4.py) and stored in the order the
waves stream it. vLLM keeps its own tensors for the prefill path, so the copies add the
linear weights once more (12.9 GB). The embedding table, the final norm and the small
per-layer tensors are read in place.
"""

import torch

from ..kernels import fp4
from ..kernels.dims import NL, is_attn
from ..kernels.gemv import CHUNKS, PAD_TILES
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


def gu_order(t: torch.Tensor) -> torch.Tensor:
    """[gate; up] rows -> per 8 columns c: gate rows 8 c .. 8 c + 7, then up rows 8 c ..
    8 c + 7, so one 16-row group of the GEMV yields 8 complete act columns."""
    n = t.shape[0] // 2
    g = t[:n].reshape(n // 8, 8, -1)
    u = t[n:].reshape(n // 8, 8, -1)
    return torch.cat([g, u], 1).reshape(2 * n, -1)


def unit_order(t: torch.Tensor, C: int) -> torch.Tensor:
    """[G, KS, ...] per-(group, superstep) tiles -> the GEMV stream's unit order
    (chunk, group, superstep), flat [C G KSc, ...], plus PAD_TILES zero tiles (loads
    run ahead)."""
    G, KS = t.shape[:2]
    u = (
        t.reshape(G, C, KS // C, *t.shape[2:])
        .transpose(0, 1)
        .reshape(G * KS, *t.shape[2:])
    )
    pad = torch.zeros(PAD_TILES, *t.shape[2:], dtype=t.dtype, device=t.device)
    return torch.cat([u, pad]).contiguous()


class PackedWeights:
    """The kernel's weights: packed GEMV tiles per (layer, GEMV) and the in-place small
    tensors per (layer, name)."""

    def __init__(self, layers):
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
                    if name == "gu":
                        w, s = gu_order(w), gu_order(s)
                    tiles, scales = fp4.repack(w, s)
                    C = CHUNKS.get(name, 1)
                    self.mats[(i, name)] = (unit_order(tiles, C), unit_order(scales, C))
            names = (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ) + SMALL[kind]
            for n in names:
                mod, _, attr = n.rpartition(".")
                t = getattr(layer.get_submodule(mod), attr)
                # the kernel reads the small tensors as bf16 (the checkpoint's dtype)
                self.small[(i, n)] = t.detach().to(torch.bfloat16).contiguous()
