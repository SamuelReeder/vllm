# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Stage helpers shared by every model's stages.py (common/DESIGN.md, M7): the x-scale
tail of a quantized GEMV input, the fp8 quantization of 8 values into LDS, and the raw
fp32 GEMV epilogue. The model prologues and epilogues (and rstd_from_ssq, D11) stay in
the models.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import range_constexpr

from . import dev as D


@D.traced
def zero_xs_tail(c, g, xs):
    """x-scale rows of tokens M..15 = 0: their MFMA rows contribute nothing."""
    n = (16 - g.M) * g.XSR
    for i in range_constexpr((n + D.NT - 1) // D.NT):
        j = c["tid"] + D.NT * i
        if j < n:
            fx.ptr_store(fx.Float32(0.0), xs + (g.M * g.XSR + j))


def store_quant8(c, g, xq, xs, x, t, k0, live):
    """8 values of token t at column k0 (16 consecutive lanes = one 128 block) ->
    LDS fp8 + the block's pow2 scale."""
    store_quant8_at(c, xq, xs, x, t, g.XR, g.XSR, k0, live)


@D.traced
def store_quant8_at(c, xq, xs, x, row, xr, xsr, k0, live):
    """store_quant8 into LDS row `row` of xr bytes (xsr scales per row)."""
    lane = c["lane"]
    am = fx.Float32(0.0)
    for e in range_constexpr(8):
        am = fx.max(am, D.absf(x[e]))
    am = D.xor_max(am, lane, (1, 2, 4, 8))
    qi, sc = D.pow2_scale(fx.max(am, fx.Float32(1e-30)))
    lo, hi = D.quant8(x, qi)
    if live:
        fx.ptr_store(
            fx.Vector.from_elements([lo, hi], fx.Int32), xq + (row * xr + k0) // 4
        )
        if k0 % 128 == 0:
            fx.ptr_store(sc, xs + (row * xsr + k0 // 128))


@D.traced
def epi_raw(c, M, N, grp, tot, out, off=0):
    """out[:, off + 16 grp + col] = tot (fp32 [M][N]). off=None leaves the offset term
    out of the index (Qwen3.8's form: `+ 0` changes its ISA in the vLLM image)."""
    ro = D.rsrc(out)
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        if t < M:
            i = t * N + grp * 16 if off is None else t * N + off + grp * 16
            D.st(tot[r], ro, i + c["col"], D.CM_DEV)
