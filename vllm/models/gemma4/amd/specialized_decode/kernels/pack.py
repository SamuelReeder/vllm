# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Packing of the FP8 checkpoint (RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic) into the
megakernel's layouts. gu_order, pack_bf16 and unit_order are copied from
qwen3.6-35b-a3b-gfx942/mk/pack.py at megakernels e2a6fb8; fp8_tiles changed to one
scale per row.

Per layer i (docs/DESIGN.md "Stages and barriers"):
  qkv    = [q_proj | k_proj | v_proj]   fp8 (sliding 8192 x 2816; full [q | k] 9216 x 2816)
  o      = o_proj                       fp8 (2816 x 4096 sliding, 2816 x 8192 full)
  router = router.proj                  bf16 (128 x 2816)
  e1     = experts 0..127, then the dense MLP as 3 pseudo-experts (columns 704 k ..
           704 k + 703 of gate_proj / up_proj), [gate | up] rows in gu_order
                                        fp8 (131 x 1408 x 2816)
  e2     = the same experts' down_proj, K padded 704 -> 768 with zeros
                                        fp8 (131 x 2816 x 768)
LM head: the tied embedding, bf16 (pack_bf16). Small tensors stay bf16.

fp8 format: e4m3fn bits read as e4m3fnuz (bias 8 instead of 7: the value halves), 0x80
(e4m3fn -0, e4m3fnuz NaN) -> 0x00, row scale x 2. Exact.
"""

from __future__ import annotations

import torch

from .gemv import PAD_TILES


def gu_order(t: torch.Tensor) -> torch.Tensor:
    """[..., gate; up rows, ...] (rows on dim -2) -> per 8 columns c: gate rows 8 c ..
    8 c + 7, then up rows 8 c .. 8 c + 7. One 16-row group holds both halves of 8 act
    columns (moe.EpiAct relies on it)."""
    *b, N2, K = t.shape
    n = N2 // 2
    g = t[..., :n, :].reshape(*b, n // 8, 8, K)
    u = t[..., n:, :].reshape(*b, n // 8, 8, K)
    return torch.cat([g, u], -2).reshape(*b, N2, K)


def gu_order_rows(s: torch.Tensor) -> torch.Tensor:
    """gu_order for per-row vectors [..., N2]."""
    return gu_order(s[..., None])[..., 0]


def fp8_tiles(w: torch.Tensor, srow: torch.Tensor):
    """e4m3fn [..., N, K] + per-row fp32 scales [..., N] -> (tiles u8 [..., G, KS, 2, 64,
    16], scales fp32 [..., G, KS, 16]).

    Lane l (row r = l % 16, gl = l // 16), load q: MFMA steps d = 2 q, 2 q + 1, 8 bytes
    each at k = 128 st + 32 d + 8 gl + 0..7. Scales: the 16 rows' scales, repeated per
    superstep (gemv.fp8_load: lane l reads row l % 16)."""
    *b, N, K = w.shape
    G, KS = N // 16, K // 128
    assert N % 16 == 0 and K % 128 == 0
    u8 = w.view(torch.uint8)
    assert not ((u8 & 0x7F) == 0x7F).any(), "e4m3fn NaN in the checkpoint"
    u8 = torch.where(u8 == 0x80, torch.zeros_like(u8), u8)
    nb = len(b)
    t = u8.reshape(*b, G, 16, KS, 2, 2, 4, 8)  # g r st q dd gl byte
    perm = list(range(nb)) + [nb + i for i in (0, 2, 3, 5, 1, 4, 6)]
    tiles = t.permute(perm).reshape(*b, G, KS, 2, 64, 16).contiguous()
    sc = (2.0 * srow.float()).reshape(*b, G, 1, 16).expand(*b, G, KS, 16).contiguous()
    return tiles, sc


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """bf16 [N, K] -> [G, KS, 4, 64, 8] bf16 (4 KiB per (group, superstep)).

    Lane l (row r = l % 16, gl = l // 16), quarter q (16 B): sub-steps s = 2q, 2q+1 of
    the 16x16x16 MFMA, each 4 bf16 at k = 128 st + 16 s + 4 gl + 0..3.
    """
    N, K = w.shape
    G, KS = N // 16, K // 128
    t = w.to(torch.bfloat16).reshape(G, 16, KS, 4, 2, 4, 4)  # g r st q h gl e
    t = t.permute(0, 2, 3, 5, 1, 4, 6)  # g st q gl r h e
    return t.reshape(G, KS, 4, 64, 8).contiguous()


def unit_order(t: torch.Tensor, nrest: int, C: int) -> torch.Tensor:
    """[..., G, KS, rest (nrest dims)] per-(group, superstep) data -> the stream's unit
    order (chunk, group, superstep), flat [(...) C G KSc, rest] (leading dims outermost,
    e.g. experts), plus PAD_TILES zero units (loads run ahead)."""
    rest = t.shape[t.dim() - nrest :]
    G, KS = t.shape[t.dim() - nrest - 2 : t.dim() - nrest]
    u = t.reshape(-1, G, C, KS // C, *rest).transpose(1, 2).reshape(-1, *rest)
    pad = torch.zeros(PAD_TILES, *rest, dtype=t.dtype, device=t.device)
    return torch.cat([u, pad]).contiguous()
