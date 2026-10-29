# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage prologues (fill the GEMV's LDS x) and last-arriver epilogues.

Activation hand-offs between stages:
  h      fp32 [M][H]       residual stream (double buffered by the caller)
  ssq    fp32 [M][H/16]    sum of squares of h per 16-column group (for RMSNorm)
  xq/xs  e4m3fnuz [M][K] + fp32 pow2 scale [M][K/128]: a quantized GEMV input
  y      fp32 [M][N]       raw GEMV output
All global hand-off traffic uses device scope (CM_DEV).
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from .dims import CONV_DIM, INP_N, VOCAB, H
from .mkcore import dev as D

NT = 512  # threads per CTA (set by the kernel builder before tracing)


@D.traced
def zero_xs_tail(c, g, xs):
    """x-scale rows of tokens M..15 = 0: their MFMA rows contribute nothing."""
    n = (16 - g.M) * g.XSR
    for i in range_constexpr((n + NT - 1) // NT):
        j = c["tid"] + NT * i
        if j < n:
            fx.ptr_store(fx.Float32(0.0), xs + (g.M * g.XSR + j))


@D.traced
def rstd_from_ssq(c, M, ssq, red, eps):
    """red[t] = 1 / sqrt(sum(ssq[t, :]) / H + eps) for t < M (wave t, fixed order)."""
    r = D.rsrc(ssq)
    lane, wave = c["lane"], c["wave"]
    ng = H // 16
    if wave < M:
        s = fx.Float32(0.0)
        for i in range_constexpr((ng + 63) // 64):
            j = lane + 64 * i
            v = D.ldf(r, wave * ng + fx.min(j, ng - 1), cm=D.CM_DEV)
            s = s + (j < ng).select(v, fx.Float32(0.0))
        s = D.wave_sum(s)  # DPP only, fixed order
        if lane == 0:
            fx.ptr_store(D.rsq(s / fx.Float32(float(H)) + fx.Float32(eps)), red + wave)
    gpu.barrier()


@D.traced
def prologue_norm(c, g, xq, xs, red, h, ssq, lnw, eps):
    """X = h * rstd * (1 + w) (zero-centred RMSNorm) for the CTA's K chunk slots,
    quantized per (token, 128 K). h: the bf16 copy of the residual (rstd comes from the
    fp32 sums of squares). The h loads go out before the rstd reduction."""
    M, K = g.M, g.K
    assert K == H and g.kind == "fp8_b128"
    tid = c["tid"]
    D.PROBE["on"] = D.PROBE.get("stage") == g.name
    D.probe(c, 5)
    zero_xs_tail(c, g, xs)
    rh, rw = D.rsrc(h), D.rsrc(lnw)
    ch0 = g.chunk0(c["bid"])
    nch = M * g.Kc // 8
    raw = []
    for sl in range_constexpr(g.slots):
        cb = fx.min(ch0 + sl, g.C - 1) * g.Kc
        for i in range_constexpr((nch + NT - 1) // NT):
            j = fx.min(tid + NT * i, nch - 1)
            t = j // (g.Kc // 8)
            kc = j % (g.Kc // 8) * 8
            k0 = cb + kc
            raw.append(
                (
                    sl,
                    i,
                    t,
                    kc,
                    D.ld(rh, (t * K + k0) // 2, 4, cm=D.CM_DEV),
                    D.ld(rw, k0 // 2, 4),
                )
            )
    D.probe(c, 6)
    rstd_from_ssq(c, M, ssq, red, eps)
    D.probe(c, 7)
    for sl, i, t, kc, hw, wv in raw:
        rs = fx.ptr_load(red + t)
        f, hv = [], []
        for e in range_constexpr(4):
            f += [D.bf_lo(wv[e]), D.bf_hi(wv[e])]
            hv += [D.bf_lo(hw[e]), D.bf_hi(hw[e])]
        x = [hv[e] * rs * (fx.Float32(1.0) + f[e]) for e in range(8)]
        store_quant8(c, g, xq, xs, x, t, sl * g.Kc + kc, tid + NT * i < nch)
    D.probe(c, 8)
    gpu.barrier()


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


class EpiRes:
    """Residual epilogue: h_out[:, 16 grp + col] = h_in + tot (fp32, and its bf16 copy
    hb_out for the next GEMV prologue); ssq_out[t][grp] = sum over the 16 columns.
    pre(grp) loads h_in; the GEMV issues it with its partial loads."""

    def __init__(self, c, M, h_in, h_out, ssq_out, hb_out):
        self.c, self.M, self.h_in, self.h_out, self.ssq_out, self.hb_out = (
            c,
            M,
            h_in,
            h_out,
            ssq_out,
            hb_out,
        )

    def pre(self, grp):
        ri = D.rsrc(self.h_in)
        return [
            D.ldf(
                ri,
                fx.min(4 * self.c["gl"] + r, self.M - 1) * H + grp * 16 + self.c["col"],
                cm=D.CM_DEV,
            )
            for r in range(4)
        ]

    def __call__(self, grp, tots, pre):
        _epi_res(
            self.c, self.M, grp, tots[0], pre, self.h_out, self.ssq_out, self.hb_out
        )


@D.traced
def _epi_res(c, M, grp, tot, hin, h_out, ssq_out, hb_out):
    lane, col = c["lane"], c["col"]
    ro, rq, rb = D.rsrc(h_out), D.rsrc(ssq_out), D.rsrc(hb_out)
    ng = H // 16
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        live = t < M
        tt = fx.min(t, M - 1)
        v = hin[r] + tot[r]
        s = D.xor_sum(v * v, lane, (1, 2, 4, 8))
        if live:
            D.st(v, ro, tt * H + grp * 16 + col, D.CM_DEV)
            D.st(fx.Float32(v).to(fx.BFloat16), rb, tt * H + grp * 16 + col, D.CM_DEV)
            if col == 0:
                D.st(s, rq, tt * ng + grp, D.CM_DEV)


class EpiConv:
    """in_proj epilogue: y[:, n] = tot for the z / b / a columns; for the q / k / v
    channels (n < CONV_DIM) the causal depthwise conv (4 taps over the 3 stored inputs
    and this one) + SiLU, and the conv state shifts by one. Row t's state is the bf16
    [3][CONV_DIM] (oldest first) at conv + sidx[t] conv_s0 bytes; slot 0 (a padding
    row) is not written. Each channel has one last arriver, so its state is read and
    written by one thread. pre(grp) loads the state and the taps with the GEMV's partial
    loads."""

    def __init__(self, c, M, y, conv, convw, sidx, conv_s0):
        self.c, self.M, self.y, self.conv, self.convw = c, M, y, conv, convw
        self.sidx, self.conv_s0 = sidx, conv_s0

    def rows(self):
        """(slot, state address) of the lane's 4 rows 4 gl + r (clamped to M - 1). The
        rows of a wave differ and their slots can lie more than 4 GB apart, so the state
        goes through per-lane global addresses."""
        rsx = D.rsrc(self.sidx)
        out = []
        for r in range_constexpr(4):
            sl = D.ld(rsx, fx.min(4 * self.c["gl"] + r, self.M - 1))
            out.append((sl, self.conv + fx.Int64(sl) * self.conv_s0))
        return out

    def pre(self, grp):
        n = fx.min(grp * 16 + self.c["col"], CONV_DIM - 1)
        rows = self.rows()
        st = []
        for sl, base in rows:
            w = [D.gld32(base + ((k * CONV_DIM + n) // 2) * 4) for k in range(3)]
            st.append([(n % 2 == 0).select(D.bf_lo(x), D.bf_hi(x)) for x in w])
        return rows, st, D.ld(D.rsrc(self.convw), n * 2, 2)

    def __call__(self, grp, tots, pre):
        _epi_conv(self.c, self.M, grp, tots[0], pre, self.y)


@D.traced
def _epi_conv(c, M, grp, tot, pre, y):
    rows, st, wv = pre
    col = c["col"]
    n = grp * 16 + col
    isconv = grp < CONV_DIM // 16
    taps = [D.bf_lo(wv[0]), D.bf_hi(wv[0]), D.bf_lo(wv[1]), D.bf_hi(wv[1])]
    ry = D.rsrc(y)
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        v = tot[r]
        acc = st[r][0] * taps[0] + st[r][1] * taps[1] + st[r][2] * taps[2] + v * taps[3]
        out = isconv.select(D.silu(acc), v)
        if t < M:
            D.st(out, ry, t * INP_N + n, D.CM_DEV)
            sl, base = rows[r]
            if isconv & (sl != 0):
                for k, x in ((0, st[r][1]), (1, st[r][2]), (2, v)):
                    D.gst_bf16(x, base + (k * CONV_DIM + n) * 2)


@D.traced
def epi_raw(c, M, N, grp, tot, out, off=0):
    """out[:, off + 16 grp + col] = tot (fp32 [M][N])."""
    ro = D.rsrc(out)
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        if t < M:
            D.st(tot[r], ro, t * N + off + grp * 16 + c["col"], D.CM_DEV)


@D.traced
def prologue_scaled(c, g, xq, xs, y, ssq, norm_eps, per_head):
    """LDS x <- y (fp32 [M][K]) for the CTA's chunk slots, quantized.
    per_head > 0: x = y * rsqrt(mean_head(o^2) + eps) with the head's 8 chunk sums of
    squares from ssq [M][K / per_head][8] (the GDN gated RMSNorm); else x = y."""
    M = g.M
    tid = c["tid"]
    zero_xs_tail(c, g, xs)
    ry, rq = D.rsrc(y), D.rsrc(ssq)
    ch0 = g.chunk0(c["bid"])
    nch = M * g.Kc // 8
    raw = []
    for sl in range_constexpr(g.slots):
        cb = fx.min(ch0 + sl, g.C - 1) * g.Kc
        for i in range_constexpr((nch + NT - 1) // NT):
            j = fx.min(tid + NT * i, nch - 1)
            t = j // (g.Kc // 8)
            kc = j % (g.Kc // 8) * 8
            k0 = cb + kc
            a = D.ld(ry, t * g.K + k0, 4, T.f32, D.CM_DEV)
            b = D.ld(ry, t * g.K + k0 + 4, 4, T.f32, D.CM_DEV)
            hs = None
            if const_expr(per_head > 0):
                hd = k0 // per_head
                hs = [
                    D.ld(rq, (t * (g.K // per_head) + hd) * 8, 4, T.f32, D.CM_DEV),
                    D.ld(rq, (t * (g.K // per_head) + hd) * 8 + 4, 4, T.f32, D.CM_DEV),
                ]
            raw.append((sl, i, t, kc, a, b, hs))
    for sl, i, t, kc, a, b, hs in raw:
        x = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
        if const_expr(per_head > 0):
            s8 = (
                hs[0][0]
                + hs[0][1]
                + hs[0][2]
                + hs[0][3]
                + hs[1][0]
                + hs[1][1]
                + hs[1][2]
                + hs[1][3]
            )
            rs = D.rsq(s8 / fx.Float32(float(per_head)) + fx.Float32(norm_eps))
            x = [v * rs for v in x]
        store_quant8(c, g, xq, xs, x, t, sl * g.Kc + kc, tid + NT * i < nch)
    gpu.barrier()


@D.traced
def prologue_norm_bf16(c, g, xq, red, h, ssq, lnw, eps):
    """LM head: LDS x <- bf16(h * rstd * (1 + w)) for the CTA's chunk slots (h: the
    bf16 copy of the residual, as in prologue_norm)."""
    M, K = g.M, g.K
    assert g.kind == "bf16"
    tid = c["tid"]
    rh, rw = D.rsrc(h), D.rsrc(lnw)
    ch0 = g.chunk0(c["bid"])
    nch = M * g.Kc // 8
    raw = []
    for sl in range_constexpr(g.slots):
        cb = fx.min(ch0 + sl, g.C - 1) * g.Kc
        for i in range_constexpr((nch + NT - 1) // NT):
            j = fx.min(tid + NT * i, nch - 1)
            t = j // (g.Kc // 8)
            kc = j % (g.Kc // 8) * 8
            k0 = cb + kc
            raw.append(
                (
                    sl,
                    i,
                    t,
                    kc,
                    D.ld(rh, (t * K + k0) // 2, 4, cm=D.CM_DEV),
                    D.ld(rw, k0 // 2, 4),
                )
            )
    D.probe(c, 6)
    rstd_from_ssq(c, M, ssq, red, eps)
    D.probe(c, 7)
    for sl, i, t, kc, hw, wv in raw:
        rs = fx.ptr_load(red + t)
        f, hv = [], []
        for e in range_constexpr(4):
            f += [D.bf_lo(wv[e]), D.bf_hi(wv[e])]
            hv += [D.bf_lo(hw[e]), D.bf_hi(hw[e])]
        x = [hv[e] * rs * (fx.Float32(1.0) + f[e]) for e in range(8)]
        words = [
            fx.Vector.from_elements([x[2 * q], x[2 * q + 1]], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
            for q in range(4)
        ]
        if tid + NT * i < nch:
            fx.ptr_store(
                fx.Vector.from_elements(words, fx.Int32),
                xq + (t * g.XR + (sl * g.Kc + kc) * 2) // 4,
            )
    gpu.barrier()


@D.traced
def embed_stage(c, M, tok_ptr, emb, h_out, ssq_out, hb_out):
    """H = embed[tok] (bf16 -> fp32) and its 16-column sums of squares. CTA b handles
    groups b, b + CUS, ... (thread: token t = tid // 16, column tid % 16)."""
    tid, bid = c["tid"], c["bid"]
    lane = c["lane"]
    ng = H // 16
    t = tid // 16
    col = tid % 16
    if t < M:
        tok = fx.min(
            fx.max(D.ld(D.rsrc(tok_ptr), t), 0), VOCAB - 1
        )  # padding rows hold anything
        re = D.rsrc(
            emb + fx.Int64(tok) * (H * 2)
        )  # the table is > 2 GiB: 64-bit row base
        ro, rq = D.rsrc(h_out), D.rsrc(ssq_out)
        for gi in range(bid, ng, c["cus"]):
            k = gi * 16 + col
            w = D.ld(re, k // 2)
            v = (k % 2 == 0).select(D.bf_lo(w), D.bf_hi(w))
            D.st(v, ro, t * H + k, D.CM_DEV)
            D.st(fx.Float32(v).to(fx.BFloat16), D.rsrc(hb_out), t * H + k, D.CM_DEV)
            s = D.xor_sum(v * v, lane, (1, 2, 4, 8))
            if col == 0:
                D.st(s, rq, t * ng + gi, D.CM_DEV)


@D.traced
def final_norm_stage(c, M, h, ssq, red, lnw, out, eps):
    """out[t][k] = bf16(h[t][k] * rstd[t] * (1 + w[k])): the model's final norm, for a
    serving engine's own LM head. CTA b handles 16-column groups b, b + CUS, ...
    (thread: token t = tid // 16, column tid % 16)."""
    rstd_from_ssq(c, M, ssq, red, eps)
    tid, bid = c["tid"], c["bid"]
    t = tid // 16
    col = tid % 16
    if t < M:
        rstd = fx.ptr_load(red + t)
        rh, rw, ro = D.rsrc(h), D.rsrc(lnw), D.rsrc(out)
        for gi in range(bid, H // 16, c["cus"]):
            k = gi * 16 + col
            w = D.ld(rw, k // 2)
            wk = (k % 2 == 0).select(D.bf_lo(w), D.bf_hi(w))
            v = D.ldf(rh, t * H + k, cm=D.CM_DEV) * rstd * (fx.Float32(1.0) + wk)
            D.st(fx.Float32(v).to(fx.BFloat16), ro, t * H + k)
