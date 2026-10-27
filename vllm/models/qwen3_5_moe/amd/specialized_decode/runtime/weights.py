# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel's copies of Qwen3.6-35B-A3B's linear weights, packed from the model.

Per layer the kernel streams (kernels/pack.py):
  GDN layer:  inp    = in_proj_qkvz                          fp8  (12288 x 2048)
              ba     = in_proj_ba                            bf16 (64 x 2048)
              out    = out_proj                              fp8  (2048 x 4096)
  attn layer: qkv    = qkv_proj                              fp8  (9216 x 2048)
              o      = o_proj                                fp8  (2048 x 4096)
  every layer router = [gate | shared_expert_gate | 0]       bf16 (272 x 2048)
              e1     = experts' w13 + shared gate_up, gu_order  fp8 (257 x 1024 x 2048)
              e2     = experts' w2 + shared down_proj        fp8  (257 x 2048 x 512)
Each is re-tiled for the gfx942 MFMA and stored in the order the waves stream it. vLLM
keeps its own tensors for the prefill path, so the copies add the linear weights once
more (about 34 GB). The embedding table, the final norm and the small per-layer tensors
are read in place (as bf16).

vLLM keeps the FP8 weights on gfx942 as e4m3fnuz: the checkpoint's e4m3fn bits with
0x80 mapped to 0 and the block scales doubled (normalize_e4m3fn_to_e4m3fnuz). The
kernel's packer starts from the checkpoint's e4m3fn bits and does the same mapping
itself, so it gets vLLM's bits as they are and vLLM's scales halved back; both steps
are exact.
"""

import torch

from ..kernels import pack
from ..kernels.dims import NE, NL, H, is_attn
from ..kernels.moe import LGN
from ..spec import expert_parts, fp8_parts

SMALL = {
    "gdn": (
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.conv1d.weight",
        "linear_attn.norm.weight",
    ),
    "attn": ("self_attn.q_norm.weight", "self_attn.k_norm.weight"),
}
GEMVS = {
    "gdn": {"inp": "linear_attn.in_proj_qkvz", "out": "linear_attn.out_proj"},
    "attn": {"qkv": "self_attn.qkv_proj", "o": "self_attn.o_proj"},
}


def checkpoint_scales(s: torch.Tensor, n: int) -> torch.Tensor:
    """Undo vLLM's doubling: e4m3fnuz block scales [.., N/128, K/128] -> the
    checkpoint's per-row e4m3fn scales [.., N, K/128] that kernels/pack.py expects."""
    return pack.row_scales(s, n) * 0.5


class PackedWeights:
    """The kernel's weights: packed GEMV (tiles, scales) per (layer, GEMV) (bf16
    GEMVs: (tiles, tiles)) and the in-place small tensors per (layer, name)."""

    def __init__(self, layers, chunks: dict[str, int]):
        self.chunks = dict(chunks)
        self.mats: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self.small: dict[tuple[int, str], torch.Tensor] = {}
        for i in range(NL):
            layer = layers[i]
            kind = "attn" if is_attn(i) else "gdn"
            for name, path in GEMVS[kind].items():
                parts = fp8_parts(layer.get_submodule(path))
                assert parts is not None, (i, path)
                w, s = parts
                self._fp8(i, name, w, checkpoint_scales(s, w.shape[0]))
            mlp = layer.mlp
            if kind == "gdn":
                self._bf16(i, "ba", layer.linear_attn.in_proj_ba.weight)
            r = torch.zeros(LGN, H, dtype=torch.bfloat16, device=mlp.gate.weight.device)
            r[:NE] = mlp.gate.weight
            r[NE] = mlp.shared_expert_gate.weight[0]
            self._bf16(i, "router", r)
            parts = expert_parts(mlp.experts)
            gu = fp8_parts(mlp.shared_expert.gate_up_proj)
            dn = fp8_parts(mlp.shared_expert.down_proj)
            assert parts is not None and gu is not None and dn is not None, i
            w13, s13, w2, s2 = parts
            u8 = lambda t: t.view(torch.uint8)  # noqa: E731
            # the shared expert is expert NE of the grouped GEMVs
            w = torch.cat([u8(w13), u8(gu[0])[None]])
            s = checkpoint_scales(torch.cat([s13, gu[1][None]]), w.shape[1])
            self._fp8(i, "e1", pack.gu_order(w), pack.gu_order(s))
            w = torch.cat([u8(w2), u8(dn[0])[None]])
            s = checkpoint_scales(torch.cat([s2, dn[1][None]]), w.shape[1])
            self._fp8(i, "e2", w, s)
            del w, s
            names = (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ) + SMALL[kind]
            for n in names:
                mod, _, attr = n.rpartition(".")
                t = getattr(layer.get_submodule(mod), attr)
                # the kernel reads the small tensors as bf16 (the checkpoint's dtype)
                self.small[(i, n)] = t.detach().to(torch.bfloat16).contiguous()

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
