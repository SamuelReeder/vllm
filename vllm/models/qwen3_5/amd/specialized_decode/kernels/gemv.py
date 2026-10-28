# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

Weights "mx": mk4 tiles (mk/fp4.py), fp8 MFMA, x fp8 with a pow2 scale per
(token, superstep). Weights "bf16": pack_bf16 tiles, bf16 MFMA, x bf16, no scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import dev as D
from . import fp4

H4 = fp4.device_helpers()
# K chunks of the GEMVs (default 1); the packer follows. chunks() gives a TP size's
# table, with an optional override ((name, C), ...): Cfg.chunks carries it, so it is in
# the kernel name.
CHUNKS = dict(out=2, o=2, down=4, lm=8)
CHUNKS_TP = {1: CHUNKS, 2: CHUNKS}


def chunks(tp=1, override=()):
    return dict(CHUNKS_TP[tp], **dict(override))


PAD_TILES = 16  # spare tiles after a packed matrix (>= 3 P - 1)


@dataclass(frozen=True)
class Gemv:
    name: str
    N: int
    K: int
    M: int
    cus: int
    waves: int = 8
    C: int = 1  # K chunks
    kind: str = "mx"  # "mx" | "bf16"
    P: int = 2  # units per pipeline block
    gpc: int = 1  # groups per counter (last-arriver granularity)

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
        return 1 if self.kind == "mx" else 2

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
    def wave_lo(self, gw):
        return (gw * self.U) // self.NW

    def wave_of(self, u):
        """Global wave whose range holds unit u."""
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
        assert max(self.nseg_host(p) for p in range(self.C * self.G)) <= self.SEGMAX
        return self

    def part_floats(self):
        return self.C * self.G * self.SEGMAX * 256

    # ------------------------------------------------------------ device
    def cta_range(self, bid):
        return (bid * self.U) // self.cus, ((bid + 1) * self.U) // self.cus

    def chunk0(self, bid):
        return self.cta_range(bid)[0] // (self.G * self.KSc)

    # ---- the stream. Tiles are packed in unit order (chunk, group, superstep), plus
    # PAD_TILES spare tiles: unit u's data is tile u, so a load needs no index math.
    def load(self, w, u):
        """Unit u's weight data for this lane."""
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
        if const_expr(self.kind == "mx"):
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
        return w

    def first_loads(self, w):
        return [self.load(w, w["u0"] + p) for p in range(self.P)]

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
        # counters after the whole stream: one drain, then all the wave's atomics at
        # once (their round trips overlap), then the reductions of the groups it
        # completed
        rocdl.s_waitcnt(vmcnt=0)
        p0 = u0 // KSc
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
                # the last arriver resets the counter for the next layer: every other
                # arrival is in, and the next use is behind a grid barrier
                if w["lane"] == 0:
                    D.st(fx.Int32(0), D.rsrc(cnt), self.cgroup(p0 + j), D.CM_DEV)
                self.la_reduce(
                    w["lane"], self.cgroup(p0 + j), rp, epilogue, c, deferred
                )
        if const_expr(deferred):
            # deferred poll: every group's push is out before the first poll
            for j in range_constexpr(self.NPW):
                if (j < np_) & ((olds[j] + 1) == self.cg_total(self.cgroup(p0 + j))):
                    self.la_finish(epilogue, self.cgroup(p0 + j))
        D.probe(c, 2)

    def nseg(self, p):
        wf = self.wave_of(p * self.KSc)
        return wf, self.wave_of((p + 1) * self.KSc - 1) - wf + 1

    @D.traced
    def store_part(self, w, p, acc, rp):
        """Piece p's partial of this wave -> slot [p][gw - first wave of p] (device
        scope)."""
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
