# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Flat unit-stream GEMV of the megakernel.

out[m, n] = sum_k x[m, k] W[n, k] for M <= 16 rows.

Units: a unit is (group of 16 output rows, superstep of 128 K). K is cut into C
chunks of KSc supersteps; units are ordered (chunk, group, superstep) and a
"piece" p = chunk * G + group is KSc consecutive units. Global wave gw streams
units [gw U / NW, (gw + 1) U / NW): one contiguous range, so every wave has the
same number of units (+-1). x lives in LDS (filled by the stage prologue) for
the <= 2 chunks a CTA touches ("slots").

A wave keeps two accumulators (the piece it is in, and the next one) and at each
piece end stores the piece's partial to slot [p][seg] (seg = gw - first wave of p).
It then adds 1 to the group's counter (device-scope atomic); the wave that makes
the count a multiple of the group's segment total sums all segments in fixed
order (deterministic) and runs the epilogue for the group ("last arriver").

Weight formats (`kind`; common/DESIGN.md D8 and "The hardest divergences" 1):
  "fp8_b128": fp8_tiles_b128 (2 KiB e4m3fnuz per unit) + 2 fp32 scales per unit (rows
              0-7, rows 8-15; 128 x 128 block scales, Qwen3.6); fp8 MFMA; x fp8 with a
              pow2 scale per (token, superstep).
  "fp8_row":  as fp8_b128 with 16 fp32 scales per unit (one per output row; per-channel
              scales, Gemma 4).
  "mx":       MXFP4 mk4 tiles (fp4.py) decoded to fp8 for the fp8 MFMA, x as fp8_b128
              (Qwen3.8).
  "bf16":     pack_bf16 tiles (4 KiB per unit), bf16 MFMA, x bf16, no scales.
Modes: `small` (U < NW: some waves own no unit; their piece count is selected to 0,
the only mode with that select), `cta` (CTA-local split-K through LDS), and the
epilogue protocol: `epilogue(cg, tots[, pre])`, optional `pre(cg)`, optional
`deferred = True` with `push(cg, tots)` and `finish(cg)` (Qwen3.8 TP2).
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import dev as D
from . import fp4
from .pack import PAD_TILES

H4 = fp4.device_helpers()


def mfma_fp8(a, b, c):
    """D = A x B + C, 16x16x32 fp8 (operands: i64 of 8 e4m3fnuz)."""
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_fp8_fp8(T.vec(4, T.f32), [a, b, c, 0, 0, 0])
    )


def _imin(a, b):
    """Return min of host ints, or of a device int and an int."""
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return fx.min(a, b)


def _fp8_acc(w, arow, t0, t1):
    """4 fp8 MFMAs (32 K each) of one unit into one accumulator."""
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for d in range_constexpr(4):
        a = fx.Vector(
            fx.ptr_load(
                w["xq"] + (arow + 32 * d) // 4,
                result_type=fx.Vector.make_type(2, fx.Int32),
            )
        ).bitcast(fx.Int64)[0]
        tq = t0 if d < 2 else t1
        b = fx.Vector.from_elements(
            [tq[2 * (d % 2)], tq[2 * (d % 2) + 1]], fx.Int32
        ).bitcast(fx.Int64)[0]
        acc = mfma_fp8(a, b, acc)
    return acc


def _fp8_scaled(w, acc, xo, xsr, wsc):
    xs = fx.Vector.from_elements(
        [fx.ptr_load(w["xs"] + ((4 * w["gl"] + r) * xsr + xo)) * wsc for r in range(4)],
        fx.Float32,
    )
    return acc * xs


def fp8_b128_load(rt, rs, u, lane):
    """Unit u of an fp8_b128 matrix: the lane's 2 x 16 B of the 2 KiB tile (load q holds
    MFMA steps 2 q, 2 q + 1) and the unit's 2 scales (rows 0-7, rows 8-15)."""
    t0 = D.ld(rt, (u * 128 + lane) * 4, 4, cm=D.CM_NT)
    t1 = D.ld(rt, (u * 128 + 64 + lane) * 4, 4, cm=D.CM_NT)
    return (t0, t1), D.ld(rs, u * 2, 2, T.f32)


def fp8_b128_compute(w, arow, xo, xsr, ops):
    """One fp8_b128 unit: 4 MFMAs into one accumulator, then x the weight scale of the
    lane's row (col < 8: rows 0-7) and the x scale of each token."""
    (t0, t1), ws = ops
    acc = _fp8_acc(w, arow, t0, t1)
    return _fp8_scaled(w, acc, xo, xsr, (w["col"] < 8).select(ws[0], ws[1]))


def fp8_row_load(rt, rs, u, lane):
    """Unit u of an fp8_row matrix: the tile as fp8_b128_load and the scale of the
    lane's row (lane % 16)."""
    t0 = D.ld(rt, (u * 128 + lane) * 4, 4, cm=D.CM_NT)
    t1 = D.ld(rt, (u * 128 + 64 + lane) * 4, 4, cm=D.CM_NT)
    return (t0, t1), D.ld(rs, u * 16 + lane % 16, 1, T.f32)


def fp8_row_compute(w, arow, xo, xsr, ops):
    """One fp8_row unit: 4 MFMAs into one accumulator, then x the weight scale of the
    lane's row and the x scale of each token."""
    (t0, t1), ws = ops
    acc = _fp8_acc(w, arow, t0, t1)
    return _fp8_scaled(w, acc, xo, xsr, ws)


FP8_KINDS = {
    "fp8_b128": (fp8_b128_load, fp8_b128_compute),
    "fp8_row": (fp8_row_load, fp8_row_compute),
}


@dataclass(frozen=True)
class Gemv:
    name: str
    N: int
    K: int
    M: int
    cus: int
    waves: int = 8
    C: int = 1  # K chunks
    kind: str = "bf16"  # "fp8_b128" | "fp8_row" | "mx" | "bf16"
    P: int = 2  # units per pipeline block
    gpc: int = 1  # groups per counter (last-arriver granularity)
    # CTA-local split-K: CTA b streams whole groups [b G / cus, (b + 1) G / cus), its
    # waves split their units evenly, and the CTA sums the wave partials in LDS (fixed
    # wave order): no global partials, no counter atomics (needs C = 1)
    cta: bool = False

    @property
    def G(self):
        return self.N // 16

    @property
    def KS(self):
        return self.K // 128

    @property
    def KSc(self):
        return self.KS // self.C

    @property
    def Kc(self):
        return self.K // self.C

    @property
    def U(self):
        return self.C * self.G * self.KSc

    @property
    def NW(self):
        return self.cus * self.waves

    @property
    def slots(self):
        """K chunks a CTA touches: 1 when C divides the CTA count (chunk boundaries fall
        on CTA boundaries), else <= 2."""
        return 1 if self.cus % self.C == 0 else 2

    @property
    def MT(self):
        """Lane groups (gl) whose accumulators hold live tokens."""
        return (self.M + 3) // 4

    @property
    def esize(self):
        return 2 if self.kind == "bf16" else 1

    @property
    def XR(self):
        """LDS bytes per x row (slots x Kc, padded 16 B)."""
        return self.slots * self.Kc * self.esize + 16

    @property
    def XSR(self):
        """LDS x-scale floats per token row."""
        return self.slots * self.KSc

    # ---- unit split: wave gw streams [gw U / NW, (gw + 1) U / NW). (Weighting the
    # waves of a CTA differently was measured: no gain; see docs/DESIGN.md.)
    # A GEMV with fewer units than waves (the router, in_proj b/a: "small") gives wave
    # gw < U unit gw instead, so a piece's segments sit in consecutive waves.
    # cta: wave w of CTA b streams [g0 KS + w n / waves, g0 KS + (w + 1) n / waves) of
    # the CTA's n = (g1 - g0) KS units, [g0, g1) = [b G / cus, (b + 1) G / cus).
    @property
    def small(self):
        return not self.cta and self.U < self.NW

    def cta_groups(self, bid):
        return (bid * self.G) // self.cus, ((bid + 1) * self.G) // self.cus

    def wave_lo(self, gw):
        if self.cta:
            b, v = gw // self.waves, gw % self.waves
            g0, g1 = self.cta_groups(b)
            return g0 * self.KS + (v * ((g1 - g0) * self.KS)) // self.waves
        if self.small:
            return _imin(gw, self.U)
        return (gw * self.U) // self.NW

    @property
    def NGC(self):
        """cta: max groups of one CTA."""
        return -(-self.G // self.cus)

    @property
    def NPWC(self):
        """cta: max pieces (groups) one wave's range touches."""
        upw = -(-self.NGC * self.KS // self.waves)  # max units of one wave
        return -(-upw // self.KS) + 1

    def lds_floats(self):
        """cta: LDS floats of the wave partials [waves][NPWC][16 MT lanes][4]."""
        return self.waves * self.NPWC * 16 * self.MT * 4 if self.cta else 0

    def wave_of(self, u):
        """Global wave whose range holds unit u."""
        if self.small:
            return u
        return ((u + 1) * self.NW - 1) // self.U

    @property
    def SEGMAX(self):
        import math

        return math.ceil(self.KSc / max(math.floor(self.U / self.NW), 1)) + 1

    def nseg_host(self, p):
        return self.wave_of((p + 1) * self.KSc - 1) - self.wave_of(p * self.KSc) + 1

    def check(self):
        assert self.N % 16 == 0 and self.K % 128 == 0 and self.KS % self.C == 0
        assert self.KSc >= self.P, "a pipeline block may cross at most one piece end"
        assert PAD_TILES >= 3 * self.P - 1, (
            "loads run up to 3 P - 1 units past the stream"
        )
        assert (self.U + 1) * self.NW < 2**31
        assert self.G % self.gpc == 0
        if self.cta:
            assert self.C == 1 and self.gpc == 1, "cta: a CTA holds whole groups, all K"
            for gw in range(self.NW):
                lo, hi = self.wave_lo(gw), self.wave_lo(gw + 1)
                assert hi == lo or (hi - 1) // self.KS - lo // self.KS < self.NPWC
            return self
        assert max(self.nseg_host(p) for p in range(self.C * self.G)) <= self.SEGMAX
        if self.small and self.C > 1:
            # a CTA's 8 units then lie in one chunk (slots == 1)
            assert (self.G * self.KSc) % self.waves == 0 and self.slots == 1
        return self

    def part_floats(self):
        return 0 if self.cta else self.C * self.G * self.SEGMAX * 256

    # ------------------------------------------------------------ device
    def cta_range(self, bid):
        if self.small:
            return _imin(bid * self.waves, self.U), _imin(
                (bid + 1) * self.waves, self.U
            )
        return (bid * self.U) // self.cus, ((bid + 1) * self.U) // self.cus

    def chunk0(self, bid):
        return self.cta_range(bid)[0] // (self.G * self.KSc)

    # ---- the stream. Tiles are packed in unit order (chunk, group, superstep), plus
    # PAD_TILES spare tiles: unit u's data is tile u, so a load needs no index math.
    def load(self, w, u):
        """Unit u's weight data for this lane."""
        if const_expr(self.kind in FP8_KINDS):
            return FP8_KINDS[self.kind][0](w["rt"], w["rs"], u, w["lane"])
        if const_expr(self.kind == "mx"):
            tile = D.ld(w["rt"], (u * 64 + w["lane"]) * 4, 4, cm=D.CM_NT)
            sw = D.ld(w["rs"], u * 16 + w["col"])
            return tile, sw
        # bf16 (LM head, > 2 GiB): a 64-bit tile base per unit (wave-uniform)
        rt = D.rsrc(w["tbase"] + fx.Int64(u) * 4096)
        return [D.ld(rt, (q * 64 + w["lane"]) * 4, 4, cm=D.CM_NT) for q in range(4)]

    def compute(self, w, xo, ops):
        """One unit's contribution (4 floats: tokens 4 gl + r, column col of its group).
        xo: the unit's x superstep in LDS (always valid: dead units read real x, and
        their result is dropped with a select; a multiply by 0 would keep NaN/inf)."""
        arow = w["arow"] + xo * (128 * self.esize)
        z4 = fx.Vector.filled(4, 0.0, fx.Float32)
        if const_expr(self.kind in FP8_KINDS):
            return FP8_KINDS[self.kind][1](w, arow, xo, self.XSR, ops)
        if const_expr(self.kind == "mx"):
            tile, sw = ops
            tacc = z4
            for d in range_constexpr(4):
                a = fx.Vector(
                    fx.ptr_load(
                        w["xq"] + (arow + 32 * d) // 4,
                        result_type=fx.Vector.make_type(2, fx.Int32),
                    )
                ).bitcast(fx.Int64)[0]
                t = H4["mfma"](a, H4["dec_dword"](tile[d]), z4)
                tacc = D.fma4(
                    t,
                    fx.Vector.filled(
                        4, H4["e8m0_f32"]((sw >> (8 * d)) & 0xFF), fx.Float32
                    ),
                    tacc,
                )
            xs = fx.Vector.from_elements(
                [
                    fx.ptr_load(w["xs"] + ((4 * w["gl"] + r) * self.XSR + xo))
                    for r in range(4)
                ],
                fx.Float32,
            )
            return tacc * xs
        acc = z4
        for q in range_constexpr(4):
            for h in range_constexpr(2):
                sub = 2 * q + h
                a = fx.Vector(
                    fx.ptr_load(
                        w["xq"] + (arow + 32 * sub) // 4,
                        result_type=fx.Vector.make_type(2, fx.Int32),
                    )
                ).bitcast(fx.Int16)
                b = fx.Vector.from_elements(
                    [ops[q][2 * h], ops[q][2 * h + 1]], fx.Int32
                ).bitcast(fx.Int16)
                acc = fx.Vector(
                    rocdl.mfma_f32_16x16x16bf16_1k(
                        T.vec(4, T.f32), [a, b, acc, 0, 0, 0]
                    )
                )
        return acc

    def setup(self, c, xq, xs, tiles, scales):
        """Per-lane constants of this GEMV. xq/xs: LDS pointers; tiles/scales: 64-bit
        global addresses of the packed weight."""
        lane, gl, col = c["lane"], c["gl"], c["col"]
        bid = c["bid"]
        w = dict(lane=lane, gl=gl, col=col, xq=xq, xs=xs, tbase=tiles)
        if const_expr(self.kind != "bf16"):
            w["rt"], w["rs"] = D.rsrc(tiles), D.rsrc(scales)
        # A operand: token = lane % 16 (clamped to M - 1; its scale row is 0 past M)
        w["arow"] = fx.min(col, self.M - 1) * self.XR + 8 * gl * self.esize // 1
        if const_expr(self.kind == "bf16"):
            w["arow"] = fx.min(col, self.M - 1) * self.XR + 4 * gl * 2
        w["ch0"] = self.chunk0(bid)
        gw = bid * self.waves + c["wave"]
        w["gw"] = gw
        w["u0"] = self.wave_lo(gw)
        w["u1"] = self.wave_lo(gw + 1)
        if const_expr(self.cta):
            w["bid"], w["wave"], w["pl"] = bid, c["wave"], c["pl"]
        return w

    def first_loads(self, w):
        u = w["u0"]
        if const_expr(self.cta):
            # an idle wave (a CTA without groups) loads tile 0 (an L2 hit) instead of
            # the next CTA's tiles
            u = (w["u1"] > u).select(u, fx.Int32(0))
        return [self.load(w, u + p) for p in range(self.P)]

    def busy(self, bid):
        """Whether CTA bid streams any unit of this GEMV (needs its LDS x)."""
        if self.cta:
            g0, g1 = self.cta_groups(bid)
            return g1 > g0
        if self.small:
            return bid * self.waves < self.U
        return bid >= 0

    @D.traced
    def run(self, c, w, buf, part, cnt, epilogue):
        """Stream this wave's units. buf: first_loads(). part/cnt: global addresses of
        the partial slots and the group counters. epilogue(cg, [acc4] * gpc[, pre]) runs
        in the last arriver of counter group cg (groups cg * gpc ...), acc4 per lane:
        tokens 4 gl + r of the group's column `col`."""
        P, KSc, G = self.P, self.KSc, self.G
        u0, u1 = w["u0"], w["u1"]
        ch0 = w["ch0"]
        XOMAX = self.slots * KSc - 1
        z4 = fx.Vector.filled(4, 0.0, fx.Float32)
        rp = D.rsrc(part)

        def add(acc, t, ok):
            return fx.Vector.from_elements(
                [ok.select(acc[r] + t[r], acc[r]) for r in range(4)], fx.Float32
            )

        @D.traced
        def block(ub, st, ops, acc_a, acc_b):
            """P units from unit ub = ((c G + g) KSc + s), st = (s, g, c). acc_a: the
            piece of ub; acc_b: the next piece (a block crosses at most one piece end).
            Blocks inside one piece and inside the stream (most) take the select-free
            path."""
            s0, g0, c0 = st
            xo0 = (c0 - ch0) * KSc + s0
            pa = c0 * G + g0
            if (s0 + P <= KSc) & (ub + P < u1):  # (the stream's last block stores)
                for p in range_constexpr(P):
                    acc_a = acc_a + self.compute(w, xo0 + p, ops[p])
                if s0 + P == KSc:
                    self.store_part(w, pa, acc_a, rp)
                    acc_a = z4
            else:
                nb = KSc - s0  # units before the piece end
                xob = ((g0 + 1 == G).select(c0 + 1, c0) - ch0) * KSc  # next piece's x
                for p in range_constexpr(P):
                    xo = fx.min((nb > p).select(xo0 + p, xob + (p - nb)), XOMAX)
                    t = self.compute(w, xo, ops[p])
                    live = ub + p < u1
                    acc_a = add(acc_a, t, live & (nb > p))
                    acc_b = add(acc_b, t, live & (nb <= p))
                if (nb <= P) | (u1 <= ub + P):
                    self.store_part(w, pa, acc_a, rp)
                    acc_a = acc_b
                    acc_b = z4
            s1 = s0 + P
            ws = s1 >= KSc
            g1 = ws.select(g0 + 1, g0)
            wg = g1 == G
            return (
                (
                    ws.select(s1 - KSc, s1),
                    wg.select(fx.Int32(0), g1),
                    wg.select(c0 + 1, c0),
                ),
                acc_a,
                acc_b,
            )

        D.PROBE["on"] = D.PROBE.get("stage") == self.name
        D.probe(c, 0)
        nblk = (u1 - u0 + P - 1) // P
        buf_a = buf
        pc0 = u0 // KSc
        st = (u0 % KSc, pc0 % G, pc0 // G)
        acc_a = z4
        acc_b = z4
        for i in range(0, (nblk + 1) // 2):
            ub = u0 + 2 * i * P
            buf_b = [self.load(w, ub + P + p) for p in range_constexpr(P)]
            st, acc_a, acc_b = block(ub, st, buf_a, acc_a, acc_b)
            buf_a = [self.load(w, ub + 2 * P + p) for p in range_constexpr(P)]
            if 2 * i + 1 < nblk:
                st, acc_a, acc_b = block(ub + P, st, buf_b, acc_a, acc_b)
        D.probe(c, 1)
        last = u1 - 1
        if ((last // KSc) != ((u0 + (nblk - 1) * P) // KSc)) & (nblk > 0):
            self.store_part(w, last // KSc, acc_a, rp)
        if const_expr(self.cta):
            # the CTA's wave partials are in LDS: one CTA barrier, then wave li % waves
            # sums the li-th group of the CTA and runs its epilogue
            gpu.barrier()
            D.probe(c, 3)
            self.cta_reduce(w, epilogue)
        else:
            # counters after the whole stream: one drain, then all the wave's atomics at
            # once (their round trips overlap), then the reductions of the groups it
            # completed
            rocdl.s_waitcnt(vmcnt=0)
            p0 = u0 // KSc
            if const_expr(self.small):
                # a wave with no units (U < NW: the router, in_proj b/a) counts no piece
                np_ = (u1 > u0).select((u1 - 1) // KSc + 1 - p0, fx.Int32(0))
            else:
                # U >= NW: every wave owns at least one unit (D8)
                np_ = (u1 - 1) // KSc + 1 - p0
            olds = []
            for j in range_constexpr(self.NPW):
                old = fx.Int32(0)
                if (w["lane"] == 0) & (j < np_):
                    old = D.atomic_add_dev(cnt + self.cgroup(p0 + j) * 4, 1)
                olds.append(D.uniform(old))
            D.probe(c, 3)
            deferred = getattr(epilogue, "deferred", False)
            for j in range_constexpr(self.NPW):
                if (j < np_) & ((olds[j] + 1) == self.cg_total(self.cgroup(p0 + j))):
                    # the last arriver resets the counter for the next layer: every
                    # other arrival is in, and the next use is behind a grid barrier
                    if w["lane"] == 0:
                        D.st(fx.Int32(0), D.rsrc(cnt), self.cgroup(p0 + j), D.CM_DEV)
                    self.la_reduce(
                        w["lane"], self.cgroup(p0 + j), rp, epilogue, c, deferred
                    )
            if const_expr(deferred):
                # deferred poll: every group's push is out before the first poll
                for j in range_constexpr(self.NPW):
                    if (j < np_) & (
                        (olds[j] + 1) == self.cg_total(self.cgroup(p0 + j))
                    ):
                        self.la_finish(epilogue, self.cgroup(p0 + j))
        D.probe(c, 2)

    def _lds_slot(self, v, j, lane):
        """cta: LDS float index of wave v's j-th piece partial, lane `lane`."""
        return ((v * self.NPWC + j) * (16 * self.MT) + lane) * 4

    @D.traced
    def cta_reduce(self, w, epilogue):
        """cta: sum each group's wave partials in wave order (each wave's own sum runs
        in unit order), then the epilogue. The order depends only on G and the CTA
        count, so the result is the same for every batch and run (P1, P2)."""
        g0, g1 = self.cta_groups(w["bid"])
        for li in range_constexpr(self.NGC):
            if (g0 + li < g1) & (w["wave"] == li % self.waves):
                self._cta_group(w, g0, g1, g0 + li, epilogue)

    @D.traced
    def _cta_group(self, w, g0, g1, gi, epilogue):
        KS, W = self.KS, self.waves
        n = (g1 - g0) * KS
        lane = fx.min(w["lane"], 16 * self.MT - 1)
        pre = epilogue.pre(gi) if hasattr(epilogue, "pre") else None
        tot = fx.Vector.filled(4, 0.0, fx.Float32)
        for v in range_constexpr(W):
            lo = g0 * KS + (v * n) // W
            hi = g0 * KS + ((v + 1) * n) // W
            hit = (hi > lo) & (lo < (gi + 1) * KS) & (hi > gi * KS)
            base = self._lds_slot(
                v, fx.min(fx.max(gi - lo // KS, 0), self.NPWC - 1), lane
            )
            tot = fx.Vector.from_elements(
                [
                    hit.select(tot[r] + fx.ptr_load(w["pl"] + (base + r)), tot[r])
                    for r in range(4)
                ],
                fx.Float32,
            )
        if const_expr(pre is None):
            epilogue(gi, [tot])
        else:
            epilogue(gi, [tot], pre)

    def nseg(self, p):
        wf = self.wave_of(p * self.KSc)
        return wf, self.wave_of((p + 1) * self.KSc - 1) - wf + 1

    @D.traced
    def store_part(self, w, p, acc, rp):
        """Piece p's partial of this wave -> slot [p][gw - first wave of p] (device
        scope); cta: -> LDS slot [wave][p - first piece of the wave]."""
        if const_expr(self.cta):
            if w["gl"] < self.MT:
                base = self._lds_slot(w["wave"], p - w["u0"] // self.KS, w["lane"])
                for r in range_constexpr(4):
                    fx.ptr_store(acc[r], w["pl"] + (base + r))
        else:
            wf, _ = self.nseg(p)
            if w["gl"] < self.MT:
                D.st(
                    acc,
                    rp,
                    ((p * self.SEGMAX + w["gw"] - wf) * 64 + w["lane"]) * 4,
                    D.CM_DEV,
                )

    @property
    def NPW(self):
        """Max pieces one wave's range touches."""
        import math

        return math.ceil(math.ceil(self.U / self.NW) / self.KSc) + 1

    def cgroup(self, p):
        return (p % self.G) // self.gpc

    def cg_total(self, cg):
        """Segments counted on counter group cg (all its pieces, all chunks)."""
        total = fx.Int32(0)
        for ch in range_constexpr(self.C):
            for j in range_constexpr(self.gpc):
                _, n = self.nseg(ch * self.G + cg * self.gpc + j)
                total = total + n
        return total

    def la_finish(self, epilogue, cg):
        """Deferred poll: the poll and the stores of a group whose push la_reduce issued
        (a method, so the dynamic if above does not carry the epilogue as state)."""
        epilogue.finish(cg)

    @D.traced
    def la_reduce(self, lane, cg, rp, epilogue, c=None, push_only=False):
        """Sum every segment of the counter group's pieces (fixed order) -> epilogue.
        A batch of groups' segment loads goes out together (dead segments masked)."""
        # the epilogue's own input loads (if any) go out with the partial loads
        pre = None
        if const_expr(not push_only and hasattr(epilogue, "pre")):
            pre = epilogue.pre(cg)
        tots = []
        nb = 4
        for j0 in range_constexpr(0, self.gpc, nb):
            pend = []
            for j in range_constexpr(j0, min(j0 + nb, self.gpc)):
                gg = cg * self.gpc + j
                lj = []
                for ch in range_constexpr(self.C):
                    pp = ch * self.G + gg
                    _, n = self.nseg(pp)
                    for sg in range_constexpr(self.SEGMAX):
                        v = D.ld(
                            rp,
                            ((pp * self.SEGMAX + fx.min(n - 1, sg)) * 64 + lane) * 4,
                            4,
                            T.f32,
                            D.CM_DEV,
                        )
                        lj.append((v, n > sg))
                pend.append(lj)
            for lj in pend:
                tot = fx.Vector.filled(4, 0.0, fx.Float32)
                for v, ok in lj:
                    tot = fx.Vector.from_elements(
                        [ok.select(tot[r] + v[r], tot[r]) for r in range(4)], fx.Float32
                    )
                tots.append(tot)
        if const_expr(push_only):
            epilogue.push(cg, tots)
        elif const_expr(pre is None):
            epilogue(cg, tots)
        else:
            epilogue(cg, tots, pre)
