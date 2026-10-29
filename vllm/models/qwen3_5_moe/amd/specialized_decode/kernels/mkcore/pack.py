# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Host-side tile layouts of the GEMV kinds (common/DESIGN.md, M6). Each model's
mk/pack.py (`Packed`, the checkpoint loader) and each vLLM runtime/weights.py pack their
tensors with these functions; the order and the byte layout follow gemv.py.

  pack_bf16       bf16 [N, K] -> [G, KS, 4, 64, 8] (kind "bf16")
  fp8_tiles_b128  e4m3fn + per-row block scales -> tiles + 2 scales per unit ("fp8_b128")
  fp8_tiles_row   e4m3fn + per-row scales -> tiles + 16 scales per unit ("fp8_row")
  row_scales      128 x 128 block scales -> per-row scales (fp8_tiles_b128 input)
  gu_order        [gate; up] rows -> 8-row interleave (the gate-up epilogues rely on it)
  gu_order_rows   gu_order for per-row vectors
  unit_order      per-(group, superstep) data -> the GEMV stream's unit order + PAD_TILES
(MXFP4 tiles of the "mx" kind: fp4.repack.)

fp8 format: e4m3fn bits read as e4m3fnuz (bias 8 instead of 7: the value halves), 0x80
(e4m3fn -0, e4m3fnuz NaN) -> 0x00, scale x 2. Exact.
"""

from __future__ import annotations

import torch

PAD_TILES = 16  # spare tiles after a packed matrix (>= 3 P - 1, gemv.Gemv asserts it)


def gu_order(t: torch.Tensor) -> torch.Tensor:
    """[..., gate; up rows, ...] (rows on dim -2) -> per 8 columns c: gate rows 8 c ..
    8 c + 7, then up rows 8 c .. 8 c + 7. One 16-row group holds both halves of 8 act
    columns (the models' gate-up epilogues rely on it)."""
    *b, N2, K = t.shape
    n = N2 // 2
    g = t[..., :n, :].reshape(*b, n // 8, 8, K)
    u = t[..., n:, :].reshape(*b, n // 8, 8, K)
    return torch.cat([g, u], -2).reshape(*b, N2, K)


def gu_order_rows(s: torch.Tensor) -> torch.Tensor:
    """gu_order for per-row vectors [..., N2]."""
    return gu_order(s[..., None])[..., 0]


def row_scales(s: torch.Tensor, N: int) -> torch.Tensor:
    """Block scales [..., N/128, K/128] -> per-row fp32 [..., N, K/128]."""
    return s.float().repeat_interleave(128, -2)[..., :N, :]


def _fp8_tiles(w: torch.Tensor):
    """e4m3fn [..., N, K] -> tiles u8 [..., G, KS, 2, 64, 16] (e4m3fnuz bits).

    Lane l (row r = l % 16, gl = l // 16), load q: MFMA steps d = 2 q, 2 q + 1, 8 bytes
    each at k = 128 st + 32 d + 8 gl + 0..7."""
    *b, N, K = w.shape
    G, KS = N // 16, K // 128
    assert N % 16 == 0 and K % 128 == 0
    u8 = w.view(torch.uint8)
    assert not ((u8 & 0x7F) == 0x7F).any(), "e4m3fn NaN in the checkpoint"
    u8 = torch.where(u8 == 0x80, torch.zeros_like(u8), u8)
    nb = len(b)
    t = u8.reshape(*b, G, 16, KS, 2, 2, 4, 8)  # g r st q dd gl byte
    perm = list(range(nb)) + [nb + i for i in (0, 2, 3, 5, 1, 4, 6)]
    return t.permute(perm).reshape(*b, G, KS, 2, 64, 16).contiguous()


def fp8_tiles_b128(w: torch.Tensor, srow: torch.Tensor):
    """Kind "fp8_b128": e4m3fn [..., N, K] + per-row fp32 scales [..., N, K/128]
    (row_scales) -> (tiles u8 [..., G, KS, 2, 64, 16], scales fp32 [..., G, KS, 2]).
    Scales: rows 0-7 and rows 8-15 of the group (each half must have one scale)."""
    *b, N, K = w.shape
    G, KS = N // 16, K // 128
    tiles = _fp8_tiles(w)
    sr = srow.reshape(*b, G, 2, 8, KS)
    assert torch.equal(sr, sr[..., :1, :].expand_as(sr)), "scale varies inside 8 rows"
    sc = (2.0 * sr[..., 0, :]).transpose(-1, -2).contiguous()  # [..., G, KS, 2]
    return tiles, sc


def fp8_tiles_row(w: torch.Tensor, srow: torch.Tensor):
    """Kind "fp8_row": e4m3fn [..., N, K] + per-row fp32 scales [..., N] -> (tiles u8
    [..., G, KS, 2, 64, 16], scales fp32 [..., G, KS, 16]). Scales: the 16 rows' scales,
    repeated per superstep (gemv.fp8_row_load: lane l reads row l % 16)."""
    *b, N, K = w.shape
    G, KS = N // 16, K // 128
    tiles = _fp8_tiles(w)
    sc = (2.0 * srow.float()).reshape(*b, G, 1, 16).expand(*b, G, KS, 16).contiguous()
    return tiles, sc


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """Kind "bf16": [N, K] (cast to bf16) -> [G, KS, 4, 64, 8] bf16 (4 KiB per (group,
    superstep)).

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
