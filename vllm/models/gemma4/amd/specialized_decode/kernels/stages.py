# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Stage prologues (fill the GEMV's LDS x) and last-arriver epilogues.

prologue_scaled and the rstd reduction are copied from qwen3.6-35b-a3b-gfx942/mk/stages.py
at megakernels e2a6fb8 (copy and diverge, docs/DESIGN.md D2); the shared helpers are in
common/mkcore/stagelib.py. The rest is Gemma 4 (docs/DESIGN.md "Stages"): every norm
after a projection needs the projection's whole row, so a consumer rebuilds the
residual from raw stage outputs ("deferred residual"):

  h1 = h + post_attention_layernorm(a)                       (router prologue)
  h' = (h1 + post_feedforward_layernorm(post_feedforward_layernorm_1(d)
             + post_feedforward_layernorm_2(e))) * layer_scalar    (next qkv prologue)

Every CTA computes the whole rows (identical instructions on identical data, so every
CTA has the same bits); CTA 0 publishes h' and h1 for the stages that need them.

Hand-off buffers (fp32, device scope):
  h, h1          [M][H]     residual at the layer start, after the attention block
  a, d, e        [M][H]     raw o_proj output, dense MLP sum, weighted expert sum
  ssq_a/d/e      [M][H/16]  their sums of squares per 16 columns (from the epilogues)
  rstd1          [M]        rstd of h1 (router prologue -> gate_up prologue)
Row layout in the whole-row prologues: thread tid < H / 8 holds columns 8 tid .. 8 tid
+ 7 of every row (16 consecutive lanes = one 128-column quantization block). A row's
values and sums do not depend on M or on the other rows.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .dims import EPS, VOCAB, H
from .mkcore import dev as D
from .mkcore.stagelib import store_quant8, zero_xs_tail

NT = 512  # threads per CTA (set by the kernel builder before tracing)
NOCT = H // 8  # threads that hold a row's columns (352)
RED_FLOATS = 8 * 8 + 32  # LDS floats of `red`: [8 waves][8 rows] partials, then rstds


@D.traced
def epi_softcap(c, M, N, grp, tot, out, cap):
    """LM head: out[:, 16 grp + col] = cap * tanh(tot / cap) (the final logit soft-cap)."""
    ro = D.rsrc(out)
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        if t < M:
            v = fx.Float32(cap) * D.tanh(tot[r] / fx.Float32(cap))
            D.st(v, ro, t * N + grp * 16 + c["col"], D.CM_DEV)


@D.traced
def _epi_ssq(c, M, grp, tot, out, ssq):
    lane, col = c["lane"], c["col"]
    ro, rq = D.rsrc(out), D.rsrc(ssq)
    for r in range_constexpr(4):
        t = 4 * c["gl"] + r
        v = tot[r]
        s = D.xor_sum(v * v, lane, (1, 2, 4, 8))
        if t < M:
            D.st(v, ro, t * H + grp * 16 + col, D.CM_DEV)
            if col == 0:
                D.st(s, rq, t * (H // 16) + grp, D.CM_DEV)


class EpiSsq:
    """o_proj epilogue: out[:, 16 grp + col] = tot (raw fp32 [M][H]) and ssq[t][grp] =
    the sum of squares of the group's 16 columns."""

    def __init__(self, c, M, out, ssq):
        self.c, self.M, self.out, self.ssq = c, M, out, ssq

    def __call__(self, grp, tots):
        _epi_ssq_t(self.c, self.M, grp, tots[0], self.out, self.ssq)


@D.traced
def _epi_ssq_t(c, M, grp, tot, out, ssq):
    _epi_ssq(c, M, grp, tot, out, ssq)


class EpiDE:
    """Expert-down epilogue (moe._reduce_down): tots = [e, d]; stores both raw with their
    16-column sums of squares. done (Cfg.combine_rows): then, if `count`, adds the group
    to the device counter `done` that the combine CTAs wait on."""

    def __init__(self, c, M, e, ssq_e, d, ssq_d, done=None, count=None):
        self.c, self.M = c, M
        self.e, self.ssq_e, self.d, self.ssq_d = e, ssq_e, d, ssq_d
        self.done, self.count = done, count

    def pre(self, grp):
        return None

    def __call__(self, grp, tots, pre):
        _epi_de(
            self.c,
            self.M,
            grp,
            tots[0],
            tots[1],
            self.e,
            self.ssq_e,
            self.d,
            self.ssq_d,
        )
        if const_expr(self.done is not None):
            _count_done(self.c, self.done, self.count)


@D.traced
def _count_done(c, done, count):
    rocdl.s_waitcnt(vmcnt=0)  # this group's stores before its count
    if count & (c["lane"] == 0):
        D.atomic_add_dev(done, 1)


@D.traced
def _epi_de(c, M, grp, te, td, e, ssq_e, d, ssq_d):
    _epi_ssq(c, M, grp, te, e, ssq_e)
    _epi_ssq(c, M, grp, td, d, ssq_d)


@D.traced
def prologue_scaled(c, g, xq, xs, y):
    """LDS x <- y (fp32 [M][K]) for the CTA's chunk slots, quantized (the o_proj input)."""
    M = g.M
    tid = c["tid"]
    _probe_on(c, g.name, 5)
    zero_xs_tail(c, g, xs)
    ry = D.rsrc(y)
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
            raw.append((sl, i, t, kc, a, b))
    for sl, i, t, kc, a, b in raw:
        x = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
        store_quant8(c, g, xq, xs, x, t, sl * g.Kc + kc, tid + NT * i < nch)
    gpu.barrier()
    D.probe(c, 8)


# ---------------------------------------------------------------- whole-row prologues


def _probe_on(c, name, slot):
    """Debug: probe stamps of the stage `name` (bench/profile_decode.py PROBE=name)."""
    D.PROBE["on"] = D.PROBE.get("stage") == name
    D.probe(c, slot)


def _oct(c):
    """(the thread's first column, whether it holds columns)."""
    tid = c["tid"]
    return fx.min(tid, NOCT - 1) * 8, tid < NOCT


def _ld8(r, i):
    a = D.ld(r, i, 4, T.f32, D.CM_DEV)
    b = D.ld(r, i + 4, 4, T.f32, D.CM_DEV)
    return [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]


def _w8(r, k0):
    """8 bf16 weights at column k0 as fp32."""
    w = D.ld(r, k0 // 2, 4)
    out = []
    for e in range_constexpr(4):
        out += [D.bf_lo(w[e]), D.bf_hi(w[e])]
    return out


@D.traced
def _rows_rstd(c, M, parts, red, slot):
    """parts[t]: the thread's partial sum of squares of row t. -> red[64 + slot * 8 + t] =
    rsqrt(sum / H + eps): the 8 waves' sums (fixed order DPP), added in wave order. All
    threads must call it (it has block barriers)."""
    lane, wave, tid = c["lane"], c["wave"], c["tid"]
    for t in range_constexpr(M):
        s = D.wave_sum(parts[t])
        if lane == 0:
            fx.ptr_store(s, red + (wave * 8 + t))
    gpu.barrier()
    if tid < M:
        tot = fx.ptr_load(red + tid)
        for wv in range_constexpr(1, NT // 64):
            tot = tot + fx.ptr_load(red + (wv * 8 + tid))
        fx.ptr_store(
            D.rsq(tot / fx.Float32(float(H)) + fx.Float32(EPS)),
            red + (64 + slot * 8 + tid),
        )
    gpu.barrier()


@D.traced
def rstd_from_ssq(c, M, ssq, red, slot):
    """red[64 + slot * 8 + t] = rsqrt(sum(ssq[t, :]) / H + eps) for t < M (wave t, fixed
    order). ssq: [M][H/16] 16-column sums of squares from an epilogue."""
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
            fx.ptr_store(
                D.rsq(s / fx.Float32(float(H)) + fx.Float32(EPS)),
                red + (64 + slot * 8 + wave),
            )


def _sq8(v, live):
    s = v[0] * v[0]
    for e in range_constexpr(1, 8):
        s = s + v[e] * v[e]
    return live.select(s, fx.Float32(0.0))


@D.traced
def rebuild_h(c, M, red, h1, d, e, ssq_d, ssq_e, w1, w2, wpf, ls, h_out, name=""):
    """The residual at a layer start (docs/DESIGN.md "Stages"):
    h = (h1 + rms(rms(d) w1 + rms(e) w2) wpf) * layer_scalar, per row in fp32 (rms(x) = x
    * rsqrt(mean(x^2) + eps)). Returns ([M] lists of the thread's 8 values, k0, live)
    and leaves rstd(h) in red[64 + 3 * 8 + t]. CTA 0 stores h to h_out. Layer 0: h1 = the
    scaled embedding, d = e = 0, w = 1, layer_scalar = 1 gives h = h1 exactly."""
    k0, live = _oct(c)
    _probe_on(c, name, 5)
    rstd_from_ssq(c, M, ssq_d, red, 0)
    rstd_from_ssq(c, M, ssq_e, red, 1)
    rd, re_, r1 = D.rsrc(d), D.rsrc(e), D.rsrc(h1)
    dv = [_ld8(rd, t * H + k0) for t in range(M)]
    ev = [_ld8(re_, t * H + k0) for t in range(M)]
    a1, a2, af = _w8(D.rsrc(w1), k0), _w8(D.rsrc(w2), k0), _w8(D.rsrc(wpf), k0)
    lsv = D.bf_lo(D.ld(D.rsrc(ls), 0))
    gpu.barrier()
    s, parts = [], []
    for t in range_constexpr(M):
        rdt = fx.ptr_load(red + (64 + t))
        ret = fx.ptr_load(red + (64 + 8 + t))
        st = [dv[t][q] * rdt * a1[q] + ev[t][q] * ret * a2[q] for q in range(8)]
        s.append(st)
        parts.append(_sq8(st, live))
    # h1 loads after d, e are consumed (register pressure at M = 8); the reduction's
    # barriers cover their latency
    hv = [_ld8(r1, t * H + k0) for t in range(M)]
    D.probe(c, 6)
    _rows_rstd(c, M, parts, red, 2)
    hh, parts = [], []
    ro = D.rsrc(h_out)
    for t in range_constexpr(M):
        rs = fx.ptr_load(red + (64 + 16 + t))
        ht = [(hv[t][q] + s[t][q] * rs * af[q]) * lsv for q in range(8)]
        hh.append(ht)
        parts.append(_sq8(ht, live))
        if (c["bid"] == 0) & live:
            D.st(fx.Vector.from_elements(ht[:4], fx.Float32), ro, t * H + k0, D.CM_DEV)
            D.st(
                fx.Vector.from_elements(ht[4:], fx.Float32),
                ro,
                t * H + k0 + 4,
                D.CM_DEV,
            )
    _rows_rstd(c, M, parts, red, 3)
    D.probe(c, 7)
    return hh, k0, live


@D.traced
def prologue_qkv(c, g, xq, xs, red, hh, k0, live, lnw):
    """Qkv GEMV input: x = h * rstd(h) * input_layernorm, quantized (g.C == 1: all K)."""
    assert g.C == 1 and g.K == H and g.kind == "fp8_row"
    M = g.M
    zero_xs_tail(c, g, xs)
    wv = _w8(D.rsrc(lnw), k0)
    for t in range_constexpr(M):
        rs = fx.ptr_load(red + (64 + 24 + t))
        x = [hh[t][q] * rs * wv[q] for q in range(8)]
        store_quant8(c, g, xq, xs, x, t, k0, live)
    gpu.barrier()
    D.probe(c, 8)


@D.traced
def combine_row(
    c, m, red, h1, d, e, ssq_d, ssq_e, w1, w2, wpf, ls, lnw, h_out, xg, xgs
):
    """Down stage, CTA cus - M + m (Cfg.combine_rows): row m of rebuild_h (the same
    arithmetic and sum orders: the same bits) -> h_out, and the next layer's qkv input
    x = h * rstd(h) * input_layernorm quantized as prologue_qkv does -> xg fp8 [M][H] and
    xgs scales [M][H / 128]."""
    k0, live = _oct(c)
    _rstd_row(c, ssq_d, red, 0, m)
    _rstd_row(c, ssq_e, red, 1, m)
    dv = _ld8(D.rsrc(d), m * H + k0)
    ev = _ld8(D.rsrc(e), m * H + k0)
    hv = _ld8(D.rsrc(h1), m * H + k0)
    a1, a2, af = _w8(D.rsrc(w1), k0), _w8(D.rsrc(w2), k0), _w8(D.rsrc(wpf), k0)
    wv = _w8(D.rsrc(lnw), k0)
    lsv = D.bf_lo(D.ld(D.rsrc(ls), 0))
    gpu.barrier()
    rdt, ret = fx.ptr_load(red + 64), fx.ptr_load(red + (64 + 8))
    s = [dv[q] * rdt * a1[q] + ev[q] * ret * a2[q] for q in range(8)]
    _rows_rstd(c, 1, [_sq8(s, live)], red, 2)
    rs = fx.ptr_load(red + (64 + 16))
    ht = [(hv[q] + s[q] * rs * af[q]) * lsv for q in range(8)]
    ro = D.rsrc(h_out)
    if live:
        D.st(fx.Vector.from_elements(ht[:4], fx.Float32), ro, m * H + k0, D.CM_DEV)
        D.st(fx.Vector.from_elements(ht[4:], fx.Float32), ro, m * H + k0 + 4, D.CM_DEV)
    _rows_rstd(c, 1, [_sq8(ht, live)], red, 3)
    rh = fx.ptr_load(red + (64 + 24))
    _quant8_g(
        c, [ht[q] * rh * wv[q] for q in range(8)], D.rsrc(xg), D.rsrc(xgs), m, k0, live
    )


@D.traced
def prologue_final(c, M, xq, xr, red, hh, k0, live, lnw, out, lm_head):
    """Final norm: x = bf16(h * rstd(h) * norm) -> LDS bf16 rows of xr bytes (LM head
    GEMV input), or (lm_head False) CTA 0 stores x to out bf16 [M][H]."""
    wv = _w8(D.rsrc(lnw), k0)
    ro = D.rsrc(out)
    for t in range_constexpr(M):
        rs = fx.ptr_load(red + (64 + 24 + t))
        x = [hh[t][q] * rs * wv[q] for q in range(8)]
        words = [D.pack_bf2_rne(x[2 * q], x[2 * q + 1]) for q in range(4)]
        if const_expr(lm_head):
            if live:
                fx.ptr_store(
                    fx.Vector.from_elements(words, fx.Int32),
                    xq + (t * xr + k0 * 2) // 4,
                )
        else:
            if (c["bid"] == 0) & live:
                D.st(fx.Vector.from_elements(words, fx.Int32), ro, (t * H + k0) // 2)
    gpu.barrier()


@D.traced
def prologue_router(
    c, g, xq, red, h, a, ssq_a, wpa, rscale, h1_out, rstd1_out, store=True
):
    """Router GEMV input (bf16, all K): h1 = h + rms(a) * post_attention_layernorm;
    x = bf16(rms(h1) * H^-0.5 * router.scale). store: CTA 0 stores h1 and rstd(h1) (else
    publish_row does)."""
    assert g.C == 1 and g.K == H and g.kind == "bf16"
    M = g.M
    k0, live = _oct(c)
    _probe_on(c, g.name, 5)
    rstd_from_ssq(c, M, ssq_a, red, 0)
    rh, ra = D.rsrc(h), D.rsrc(a)
    hv = [_ld8(rh, t * H + k0) for t in range(M)]
    av = [_ld8(ra, t * H + k0) for t in range(M)]
    wv = _w8(D.rsrc(wpa), k0)
    sv = _w8(D.rsrc(rscale), k0)
    gpu.barrier()
    h1, parts = [], []
    ro = D.rsrc(h1_out)
    for t in range_constexpr(M):
        rat = fx.ptr_load(red + (64 + t))
        ht = [hv[t][q] + av[t][q] * rat * wv[q] for q in range(8)]
        h1.append(ht)
        parts.append(_sq8(ht, live))
        if const_expr(store):  # noqa: SIM102 (trace-time test, then a run-time one)
            if (c["bid"] == 0) & live:
                D.st(
                    fx.Vector.from_elements(ht[:4], fx.Float32),
                    ro,
                    t * H + k0,
                    D.CM_DEV,
                )
                D.st(
                    fx.Vector.from_elements(ht[4:], fx.Float32),
                    ro,
                    t * H + k0 + 4,
                    D.CM_DEV,
                )
    _rows_rstd(c, M, parts, red, 1)
    D.probe(c, 7)
    cinv = fx.Float32(float(H) ** -0.5)
    for t in range_constexpr(M):
        rs = fx.ptr_load(red + (64 + 8 + t))
        x = [h1[t][q] * rs * cinv * sv[q] for q in range(8)]
        words = [D.pack_bf2_rne(x[2 * q], x[2 * q + 1]) for q in range(4)]
        if live:
            fx.ptr_store(
                fx.Vector.from_elements(words, fx.Int32), xq + (t * g.XR + k0 * 2) // 4
            )
    if const_expr(store):  # noqa: SIM102 (trace-time test, then a run-time one)
        if (c["bid"] == 0) & (c["tid"] < M):
            D.st(
                fx.ptr_load(red + (64 + 8 + c["tid"])),
                D.rsrc(rstd1_out),
                c["tid"],
                D.CM_DEV,
            )
    gpu.barrier()
    D.probe(c, 8)


@D.traced
def _rstd_row(c, ssq, red, slot, m):
    """red[64 + slot * 8] = rsqrt(sum(ssq[m, :]) / H + eps) (wave 0, the order of
    rstd_from_ssq: the same bits as its row m)."""
    r = D.rsrc(ssq)
    lane = c["lane"]
    ng = H // 16
    if c["wave"] == 0:
        s = fx.Float32(0.0)
        for i in range_constexpr((ng + 63) // 64):
            j = lane + 64 * i
            v = D.ldf(r, m * ng + fx.min(j, ng - 1), cm=D.CM_DEV)
            s = s + (j < ng).select(v, fx.Float32(0.0))
        s = D.wave_sum(s)
        if lane == 0:
            fx.ptr_store(
                D.rsq(s / fx.Float32(float(H)) + fx.Float32(EPS)), red + (64 + slot * 8)
            )


@D.traced
def _quant8_g(c, x, rq, rsc, row, k0, live):
    """store_quant8_at into global memory: 8 bytes at row * H + k0 of rq, the block's
    scale at row * (H / 128) + k0 / 128 of rsc (the LDS image of the gate_up input)."""
    lane = c["lane"]
    am = fx.Float32(0.0)
    for e in range_constexpr(8):
        am = fx.max(am, D.absf(x[e]))
    am = D.xor_max(am, lane, (1, 2, 4, 8))
    qi, sc = D.pow2_scale(fx.max(am, fx.Float32(1e-30)))
    lo, hi = D.quant8(x, qi)
    if live:
        D.st(
            fx.Vector.from_elements([lo, hi], fx.Int32),
            rq,
            (row * H + k0) // 4,
            D.CM_DEV,
        )
        if k0 % 128 == 0:
            D.st(sc, rsc, row * (H // 128) + k0 // 128, D.CM_DEV)


@D.traced
def publish_row(c, M, m, red, h, a, ssq_a, wpa, w_e, w_d, h1_out, rstd1_out, xg, xgs):
    """Router stage, CTA NR + m (docs/DESIGN.md "Optimization log"): row m's h1 = h +
    rms(a) * post_attention_layernorm (the router prologue's arithmetic for that row) ->
    h1_out, rstd(h1) -> rstd1_out, and the gate_up inputs quantized: row m of x_e =
    rms(h1) * pre_feedforward_layernorm_2 and row M + m of x_d = rms(h1) *
    pre_feedforward_layernorm -> xg (fp8 [2M][H]) and xgs (scales [2M][H / 128])."""
    k0, live = _oct(c)
    _rstd_row(c, ssq_a, red, 0, m)
    rh, ra = D.rsrc(h), D.rsrc(a)
    hv = _ld8(rh, m * H + k0)
    av = _ld8(ra, m * H + k0)
    wv = _w8(D.rsrc(wpa), k0)
    we, wd = _w8(D.rsrc(w_e), k0), _w8(D.rsrc(w_d), k0)
    gpu.barrier()
    rat = fx.ptr_load(red + 64)
    ht = [hv[q] + av[q] * rat * wv[q] for q in range(8)]
    ro = D.rsrc(h1_out)
    if live:
        D.st(fx.Vector.from_elements(ht[:4], fx.Float32), ro, m * H + k0, D.CM_DEV)
        D.st(fx.Vector.from_elements(ht[4:], fx.Float32), ro, m * H + k0 + 4, D.CM_DEV)
    _rows_rstd(c, 1, [_sq8(ht, live)], red, 1)
    rs = fx.ptr_load(red + (64 + 8))
    if c["tid"] == 0:
        D.st(rs, D.rsrc(rstd1_out), m, D.CM_DEV)
    n = [ht[q] * rs for q in range(8)]
    rq, rsc = D.rsrc(xg), D.rsrc(xgs)
    _quant8_g(c, [n[q] * we[q] for q in range(8)], rq, rsc, m, k0, live)
    _quant8_g(c, [n[q] * wd[q] for q in range(8)], rq, rsc, M + m, k0, live)


@D.traced
def prologue_gu_copy(c, g, xq, xs, xg, xgs, rows=0):
    """GEMV input published to global memory (publish_row: the gate_up input, 2M rows;
    publish_qkv: the qkv input, M rows): copy the fp8 rows [rows][H] and their scales
    into the LDS layout the GEMV reads (g.C == 1); the other scale rows are 0."""
    M = g.M
    assert g.Kc == H and g.slots == 1
    rows = rows or 2 * M
    tid = c["tid"]
    _probe_on(c, g.name, 5)
    nz = 32 * g.XSR
    for i in range_constexpr((nz + NT - 1) // NT):
        k = tid + NT * i
        if k < nz:
            fx.ptr_store(fx.Float32(0.0), xs + k)
    gpu.barrier()
    rq, rsc = D.rsrc(xg), D.rsrc(xgs)
    per = H // 16  # 16-byte vectors per row
    nv = rows * per
    vals = []
    for i in range_constexpr((nv + NT - 1) // NT):
        k = fx.min(tid + NT * i, nv - 1)
        vals.append(
            (i, k, D.ld(rq, (k // per * H + k % per * 16) // 4, 4, cm=D.CM_DEV))
        )
    ns = rows * (H // 128)
    svals = []
    for i in range_constexpr((ns + NT - 1) // NT):
        k = fx.min(tid + NT * i, ns - 1)
        svals.append((i, k, D.ldf(rsc, k, cm=D.CM_DEV)))
    for i, k, v in vals:
        if tid + NT * i < nv:
            fx.ptr_store(v, xq + (k // per * g.XR + k % per * 16) // 4)
    for i, k, v in svals:
        if tid + NT * i < ns:
            fx.ptr_store(v, xs + (k // (H // 128) * g.XSR + k % (H // 128)))
    gpu.barrier()
    D.probe(c, 8)


@D.traced
def prologue_gu(c, g, xq, xs, red, h1, rstd1, w_e, w_d):
    """Expert gate_up input: rows t < M: rms(h1) * pre_feedforward_layernorm_2 (routed
    experts); rows M + t: rms(h1) * pre_feedforward_layernorm (the dense MLP), each
    quantized per (row, 128 K). rstd(h1) comes from the router prologue."""
    M = g.M
    tid = c["tid"]
    k0, live = _oct(c)
    _probe_on(c, g.name, 5)
    nz = 32 * g.XSR
    for i in range_constexpr((nz + NT - 1) // NT):
        k = tid + NT * i
        if k < nz:
            fx.ptr_store(fx.Float32(0.0), xs + k)
    gpu.barrier()
    r1 = D.rsrc(h1)
    hv = [_ld8(r1, t * H + k0) for t in range(M)]
    rs = [D.ldf(D.rsrc(rstd1), t, cm=D.CM_DEV) for t in range(M)]
    we, wd = _w8(D.rsrc(w_e), k0), _w8(D.rsrc(w_d), k0)
    for t in range_constexpr(M):
        n = [hv[t][q] * rs[t] for q in range(8)]
        store_quant8(c, g, xq, xs, [n[q] * we[q] for q in range(8)], t, k0, live)
        store_quant8(c, g, xq, xs, [n[q] * wd[q] for q in range(8)], M + t, k0, live)
    gpu.barrier()
    D.probe(c, 8)


@D.traced
def embed_stage(c, M, tok_ptr, emb, scale, h1, d, e, ssq_d, ssq_e):
    """Layer 0's rebuild inputs: h1 = bf16(embed[tok] * scale) (bf16 x bf16 as vLLM and
    HF compute it, round to nearest even), d = e = 0 and their sums of squares 0. CTA b
    handles 16-column groups b, b + CUS, ... (thread: token t = tid // 16, column tid %
    16)."""
    tid, bid = c["tid"], c["bid"]
    ng = H // 16
    t = tid // 16
    col = tid % 16
    if t < M:
        tok = fx.min(fx.max(D.ld(D.rsrc(tok_ptr), t), 0), VOCAB - 1)  # padding rows
        re = D.rsrc(emb + fx.Int64(tok) * (H * 2))  # the table is > 2 GiB
        r1, rd, re2 = D.rsrc(h1), D.rsrc(d), D.rsrc(e)
        rqd, rqe = D.rsrc(ssq_d), D.rsrc(ssq_e)
        for gi in range(bid, ng, c["cus"]):
            k = gi * 16 + col
            w = D.ld(re, k // 2)
            v = (k % 2 == 0).select(D.bf_lo(w), D.bf_hi(w)) * fx.Float32(scale)
            v = (D.bf16_rne(v) << 16).bitcast(fx.Float32)
            D.st(v, r1, t * H + k, D.CM_DEV)
            D.st(fx.Float32(0.0), rd, t * H + k, D.CM_DEV)
            D.st(fx.Float32(0.0), re2, t * H + k, D.CM_DEV)
            if col == 0:
                D.st(fx.Float32(0.0), rqd, t * ng + gi, D.CM_DEV)
                D.st(fx.Float32(0.0), rqe, t * ng + gi, D.CM_DEV)
