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

from vllm.models.deepseek_v41.amd.mono.common.sync import Mailbox, preg

from . import tpdev as TD
from .dims import VOCAB, H
from .mkcore import dev as D
from .mkcore.stagelib import store_quant8, zero_xs_tail

NT = 512  # threads per CTA (set by the kernel builder before tracing)


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
        s = D.xor_sum(s, lane, (32, 16, 8, 4, 2, 1))
        if lane == 0:
            fx.ptr_store(D.rsq(s / fx.Float32(float(H)) + fx.Float32(eps)), red + wave)
    gpu.barrier()


@D.traced
def prologue_norm(c, g, xq, xs, red, h, ssq, lnw, eps):
    """X = h * rstd * (1 + w) (zero-centred RMSNorm) for the CTA's K chunk slots,
    quantized per (token, 128 K). h: the bf16 copy of the residual (rstd comes from the
    fp32 sums of squares). The h loads go out before the rstd reduction."""
    M, K = g.M, g.K
    assert K == H and g.kind == "mx"
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


@D.traced
def prologue_act(c, g, xq, xs, act):
    """LDS x <- act (fp32 [M][FFN], written by EpiAct) for the CTA's chunk slots,
    quantized. All loads go out first."""
    M = g.M
    tid = c["tid"]
    zero_xs_tail(c, g, xs)
    ra = D.rsrc(act)
    ch0 = g.chunk0(c["bid"])
    nch = M * g.Kc // 8
    raw = []
    for sl in range_constexpr(g.slots):
        ch = fx.min(ch0 + sl, g.C - 1)
        for i in range_constexpr((nch + NT - 1) // NT):
            j = fx.min(tid + NT * i, nch - 1)
            t = j // (g.Kc // 8)
            kc = j % (g.Kc // 8) * 8  # column within the chunk
            k = t * g.K + ch * g.Kc + kc
            raw.append(
                (sl, i, t, kc, [D.ld(ra, k + o, 4, T.f32, D.CM_DEV) for o in (0, 4)])
            )
    for sl, i, t, kc, v in raw:
        x = [v[0][0], v[0][1], v[0][2], v[0][3], v[1][0], v[1][1], v[1][2], v[1][3]]
        store_quant8(c, g, xq, xs, x, t, sl * g.Kc + kc, tid + NT * i < nch)
    gpu.barrier()


class EpiRes:
    """Residual epilogue: h_out[:, 16 grp + col] = h_in + tot (fp32, and its bf16 copy
    hb_out for the next GEMV prologue); ssq_out[t][grp] = sum over the 16 columns.
    pre(grp) loads h_in; the GEMV issues it with its partial loads.

    TP > 1 (c["ar"] set): tot is this rank's partial; ar = (tag, half) of this
    all-reduce. The GEMV calls push(grp, tots) for every group it completes, then
    finish(grp): tot becomes the ranks' partials summed in rank order (ar_push,
    ar_poll; docs/DESIGN-TP2.md sections 5, 6, 11)."""

    def __init__(self, c, M, h_in, h_out, ssq_out, hb_out, ar=None):
        self.c, self.M, self.h_in, self.h_out, self.ssq_out, self.hb_out = (
            c,
            M,
            h_in,
            h_out,
            ssq_out,
            hb_out,
        )
        self.ar = ar if TD.TRACE["ar"] is not None else None
        # deferred poll: the GEMV pushes every group it completes first (push), then
        # polls and stores them (finish), so the round trips of a wave's groups overlap
        self.deferred = self.ar is not None and not TD.TRACE["ar"]["local"]

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
        tot = tots[0]
        if const_expr(self.ar is not None):  # ar_local only (deferred otherwise)
            tot = ar_local(self.c, self.M, grp, tot, *self.ar)
        _epi_res(self.c, self.M, grp, tot, pre, self.h_out, self.ssq_out, self.hb_out)

    def push(self, grp, tots):
        ar_push(self.c, self.M, grp, tots[0], *self.ar)

    def finish(self, grp):
        pre = self.pre(grp)
        tot = ar_poll(self.c, self.M, grp, *self.ar)
        _epi_res(self.c, self.M, grp, tot, pre, self.h_out, self.ssq_out, self.hb_out)


def _ar_ctx(c, grp, tag, half):
    A = dict(TD.TRACE["ar"], **c["ar"])
    off = fx.Int64(half * A["region"])
    dst = [preg(A["bases"][p] + off, 0, "ar") for p in range(A["tp"])]
    own = preg(A["own"] + off, 0, "ar")
    rows = [4 * c["gl"] + r for r in range(4)]
    k = ((fx.Int32(tag) - A["epoch"] - 1) * c["cus"] * 8 + c["bid"] * 8 + c["wave"]) * 4
    return A, dst, own, rows, grp * 16 + c["col"], k


@D.traced
def ar_push(c, M, grp, tot, tag, half):
    """This rank's partial tot[r] (row 4 gl + r, column 16 grp + col) to region `half`
    of every rank's peer buffer, its own included: [src rank][M][H] (value, tag) pairs
    ("fp32") or [src][M][H / 2] (bf16 pair, tag) words ("bf16")."""
    A, dst, own, rows, n, k = _ar_ctx(c, grp, tag, half)
    tp, rank, lane, col = A["tp"], A["rank"], c["lane"], c["col"]
    tv = [fx.Float32(tot[r]) for r in range(4)]
    tag = fx.Int32(tag)
    apw = TD.ARPROBE["wave"]
    if const_expr(apw is not None):
        t_push = D.memrealtime()
    if const_expr(A["dtype"] == "fp32"):
        # 16-byte pushes: the even lane stores its pair and its odd neighbour's
        even = col % 2 == 0
        for r in range_constexpr(4):
            nb = D.gather_f(tv[r], lane ^ 1)
            if (rows[r] < M) & even:
                for p in range_constexpr(tp):
                    Mailbox(tag).put_words(
                        dst[p],
                        (rank * M + rows[r]) * H + n,
                        [tv[r].bitcast(fx.Int32), fx.Float32(nb).bitcast(fx.Int32)],
                    )
    else:
        even = col % 2 == 0
        for r in range_constexpr(4):
            nb = D.gather_f(tv[r], lane ^ 1)
            if (rows[r] < M) & even:
                for p in range_constexpr(tp):
                    Mailbox(tag).put_bf(
                        dst[p],
                        2 * ((rank * M + rows[r]) * (H // 2) + n // 2),
                        [tv[r], nb],
                    )
    if const_expr(apw is not None):
        _stamps(apw, lane, [(k, t_push, t_push)])


@D.traced
def ar_poll(c, M, grp, tag, half):
    """Every rank's partial from the own region (the own one included, pushed by
    ar_push), summed in rank order: every rank adds the same bits in the same order,
    so all ranks agree."""
    A, dst, own, rows, n, k = _ar_ctx(c, grp, tag, half)
    tp, lane, col = A["tp"], c["lane"], c["col"]
    mb = Mailbox(fx.Int32(tag))
    apw = TD.ARPROBE["wave"]
    fp = A["dtype"] == "fp32"
    W = H if fp else H // 2
    even = col % 2 == 0
    if const_expr(fp):  # the even column's 16-byte pair of pairs
        specs = [
            (own, (s * M + fx.min(rows[r], M - 1)) * W + n - col % 2, 2)
            for r in range(4)
            for s in range(tp)
        ]
    else:
        specs = [
            (own, (s * M + fx.min(rows[r], M - 1)) * W + (n if fp else n // 2), 1)
            for r in range(4)
            for s in range(tp)
        ]
    if const_expr(apw is not None):
        t_ps = D.memrealtime()
    ws = mb.poll(specs)
    if const_expr(apw is not None):
        t_pe = D.memrealtime()
    out = []
    for r in range_constexpr(4):
        if const_expr(fp):
            vals = [
                fx.Int32(even.select(ws[r * tp + s][0], ws[r * tp + s][1])).bitcast(
                    fx.Float32
                )
                for s in range(tp)
            ]
        else:
            vals = [
                even.select(D.bf_lo(ws[r * tp + s][0]), D.bf_hi(ws[r * tp + s][0]))
                for s in range(tp)
            ]
        a = vals[0]
        for s in range_constexpr(1, tp):
            a = a + vals[s]
        out.append(a)
    if const_expr(apw is not None):
        _stamps(apw, lane, [(k + 1, t_ps, t_pe), (k + 2, t_pe, fx.Int64(grp))])
    return fx.Vector.from_elements(out, fx.Float32)


@D.traced
def _stamps(apw, lane, pairs):
    """Debug (dev.ARPROBE): lane 0 stores the i64 pairs (index, a, b) of a wave."""
    if lane == 0:
        for i, a, b in pairs:
            TD.st_stamps(fx.Int64(apw), i, a, b)


@D.traced
def ar_local(c, M, grp, tot, tag, half):
    """Timing-only stand-in of the all-reduce (Cfg.ar_local): the own partial added
    twice, no peer traffic, the same probe stamps as ar_push / ar_poll."""
    A, _, _, _, _, k = _ar_ctx(c, grp, tag, half)
    apw = TD.ARPROBE["wave"]
    if const_expr(apw is not None):
        t_push = D.memrealtime()
    out = []
    for r in range_constexpr(4):
        v = D.bf16r(tot[r]) if A["dtype"] == "bf16" else fx.Float32(tot[r])
        out.append(v + v)
    if const_expr(apw is not None):
        t_ps = D.memrealtime()
        _stamps(apw, c["lane"], [(k, t_push, t_ps), (k + 2, t_ps, fx.Int64(grp))])
    return fx.Vector.from_elements(out, fx.Float32)


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
        CONV_DIM = TD.TRACE["d"].conv_dim
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
    CONV_DIM, INP_N = TD.TRACE["d"].conv_dim, TD.TRACE["d"].inp_n
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
def epi_act(c, M, grp, tot, act):
    """gate_up epilogue: group grp = gate rows then up rows of act columns 8 grp .. + 7
    (pack.gu_order): act[:, 8 grp + col] = silu(gate) * up for lanes col < 8."""
    col, lane = c["col"], c["lane"]
    FFN = TD.TRACE["d"].ffn
    ra = D.rsrc(act)
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        up = D.gather_f(tot[r], lane + 8)  # lane col + 8 holds the up row
        if (t < M) & (col < 8):
            D.st(D.silu(tot[r]) * up, ra, t * FFN + grp * 8 + col, D.CM_DEV)


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
def embed_stage(c, M, tok_ptr, emb, h_out, ssq_out, hb_out, tag):
    """H = embed[tok] (bf16 -> fp32) and its 16-column sums of squares. CTA b handles
    groups b, b + CUS, ... (thread: token t = tid // 16, column tid % 16).

    TP > 1: emb is this rank's vocab slice. The rank that owns the token puts the row
    to the embedding region of every rank (tag `tag`), and every rank takes it from
    the owner's slot: the exact row, as at TP 1 (docs/DESIGN-TP2.md section 5)."""
    tid, bid = c["tid"], c["bid"]
    lane = c["lane"]
    d = TD.TRACE["d"]
    ng = H // 16
    t = tid // 16
    col = tid % 16
    if t < M:
        tok = fx.min(
            fx.max(D.ld(D.rsrc(tok_ptr), t), 0), VOCAB - 1
        )  # padding rows hold anything
        ro, rq = D.rsrc(h_out), D.rsrc(ssq_out)
        row = tok
        owner = fx.Int32(0)
        if const_expr(TD.TRACE["ar"] is not None):
            owner = tok // d.vocab
            row = tok - owner * d.vocab
        re = D.rsrc(
            emb + fx.Int64(row) * (H * 2)
        )  # the table is > 2 GiB: 64-bit row base
        for gi in range(bid, ng, c["cus"]):
            k = gi * 16 + col
            w = D.ld(re, k // 2)
            v = (k % 2 == 0).select(D.bf_lo(w), D.bf_hi(w))
            # ar_local (timing only) skips the exchange too: with no peer traffic in
            # the layers, the ranks do not keep in step, and the one embedding region
            # would be overwritten before it is read. The non-owner rank then reads a
            # wrong row.
            if const_expr(TD.TRACE["ar"] is not None and not TD.TRACE["ar"]["local"]):
                v = _embed_exchange(c, M, tag, owner, t, k, v)
            D.st(v, ro, t * H + k, D.CM_DEV)
            D.st(fx.Float32(v).to(fx.BFloat16), D.rsrc(hb_out), t * H + k, D.CM_DEV)
            s = D.xor_sum(v * v, lane, (1, 2, 4, 8))
            if col == 0:
                D.st(s, rq, t * ng + gi, D.CM_DEV)


@D.traced
def _embed_exchange(c, M, tag, owner, t, k, v):
    """The owner rank puts its row value v to every rank's embedding region; every
    rank returns the owner's value from its own region."""
    A = dict(TD.TRACE["ar"], **c["ar"])
    off = fx.Int64(2 * A["region"])
    mb = Mailbox(tag)
    if owner == A["rank"]:
        for p in range_constexpr(A["tp"]):
            Mailbox(tag).put(
                preg(A["bases"][p] + off, 0, "embed"),
                (owner * M + t) * H + k,
                fx.Float32(v),  # an f32 put keeps the bits (else Int32(v) converts)
            )
    w = mb.poll([(preg(A["own"] + off, 0, "embed"), (owner * M + t) * H + k, 1)])
    return fx.Int32(w[0][0]).bitcast(fx.Float32)


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
