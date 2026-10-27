# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 weight layout and device helpers for gfx942 (no MX MFMA).

Math: v_mfma_f32_16x16x32_fp8_fp8 (e4m3fnuz). A = activations (row = token),
B = weights (column = output row). e2m1 codes decode to e4m3fnuz exactly with one
v_perm_b32 byte lookup per 4 values. The e8m0 block scale (32 K) multiplies the
fp32 MFMA partial of its block: one MFMA covers exactly one scale block.

Layout ("mk4"): weight [N, K] (N % 16 == 0, K % 128 == 0) -> groups of 16 rows x
supersteps of 128 K. One (group, superstep) = one 1 KiB tile: lane l owns 16 B,
row r = l % 16, gl = l // 16. Dword d (0..3) of the lane = MFMA step d (K block
4 st + d): byte b low nibble = k 128 st + 32 d + 8 gl + b, high nibble = ... + 4 + b.
Scales: one dword per (group, superstep, row) = the row's 4 e8m0 bytes of blocks
4 st .. 4 st + 3 (byte d = block d). Scale layout [group][superstep][16 rows].
Negative zero (code 8) is stored as code 0 (0x80 is NaN in e4m3fnuz).
"""

from __future__ import annotations

import torch

STEP = 128  # K of one superstep
ROWS = 16  # rows of one group
# e2m1 magnitude 0..7 -> e4m3fnuz byte (bias 8): 0, .5, 1, 1.5, 2, 3, 4, 6
FP8_LUT = (0x00, 0x38, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54)


def repack(w_u8: torch.Tensor, s_u8: torch.Tensor):
    """Quark [N, K/2] (low nibble = even k) + [N, K/32] -> (tiles u8, scales i32).

    tiles: [G, KS, 64, 16] uint8; scales: [G, KS, 16] int32.
    """
    N, K2 = w_u8.shape
    K = 2 * K2
    assert N % ROWS == 0 and K % STEP == 0, (N, K)
    G, KS = N // ROWS, K // STEP
    nib = torch.stack([w_u8 & 0xF, w_u8 >> 4], -1).reshape(N, K)
    nib = torch.where(nib == 8, torch.zeros_like(nib), nib)
    # k = 128 st + 32 d + 8 gl + 4 h + b
    nib = nib.reshape(G, ROWS, KS, 4, 4, 2, 4)  # g r st d gl h b
    nib = nib.permute(0, 2, 4, 1, 3, 6, 5)  # g st gl r d b h
    byte = nib[..., 0] | (nib[..., 1] << 4)
    tiles = byte.reshape(G, KS, 64, 16).contiguous()
    sc = s_u8.reshape(G, ROWS, KS, 4).permute(0, 2, 1, 3).contiguous()
    return tiles, sc.view(torch.int32).reshape(G, KS, ROWS)


# ---------------------------------------------------------------- device side
def device_helpers():
    """FlyDSL helpers (imported lazily: needs flydsl)."""
    import flydsl.expr as fx
    from flydsl.expr import rocdl
    from flydsl.expr.typing import T

    T_LO = FP8_LUT[0] | FP8_LUT[1] << 8 | FP8_LUT[2] << 16 | FP8_LUT[3] << 24
    T_HI = FP8_LUT[4] | FP8_LUT[5] << 8 | FP8_LUT[6] << 16 | FP8_LUT[7] << 24
    SIGN = -0x7F7F7F80  # 0x80808080

    def perm(hi, lo, sel):
        return fx.Int32(rocdl.perm_b32(fx.Int32(hi), fx.Int32(lo), fx.Int32(sel)))

    def dec_dword(w):
        """One storage dword (8 e2m1) -> i64 of 8 e4m3fnuz (MFMA B operand)."""
        lo = perm(T_HI, T_LO, w & 0x07070707) | ((w << 4) & SIGN)
        hi = perm(T_HI, T_LO, (w >> 4) & 0x07070707) | (w & SIGN)
        return fx.Vector.from_elements([lo, hi], fx.Int32).bitcast(fx.Int64)[0]

    def mfma(a, b, c):
        """D = A x B + C, 16x16x32 fp8 (operands: i64 of 8 e4m3fnuz)."""
        return fx.Vector(
            rocdl.mfma_f32_16x16x32_fp8_fp8(T.vec(4, T.f32), [a, b, c, 0, 0, 0])
        )

    def e8m0_f32(code):
        """e8m0 byte (1..254) -> 2^(code-127) as f32."""
        return (fx.Int32(code) << 23).bitcast(fx.Float32)

    def fp8_pack4(a, b, c, d):
        """Four f32 (in range) -> one dword of four e4m3fnuz, RNE."""
        lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
        return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))

    return dict(
        dec_dword=dec_dword,
        mfma=mfma,
        e8m0_f32=e8m0_f32,
        perm=perm,
        fp8_pack4=fp8_pack4,
    )
