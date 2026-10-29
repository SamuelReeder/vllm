# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE stages of the megakernel (docs/DESIGN.md, "MoE stage design").

S4 (router) is a dense bf16 GEMV (gemv.Gemv) writing logits [M][LGN] fp32: columns
0..255 the router, 256 the shared-expert gate.

routing(): every CTA, after the S4 barrier, builds in its LDS (Route):
  pick[m][j], w[m][j]  row m's j-th expert (rank order) and weight; w[m][8] = the shared
                       gate sigmoid. Padding rows (rowv[m] == 0) pick -1.
  eid[p], npos         the positions: the selected expert ids ascending, then NE (the
                       shared expert, packed as expert NE of each stack).
  slot[m][p]           the slot j of row m for position p, or -1 (shared: 8).

MoeGemv: grouped fp8 GEMV over the npos positions. A piece is (position, chunk c, group
g) of KSp units; pieces are ordered (position, chunk, group) and global wave gw computes
pieces [gw T / NW, (gw + 1) T / NW) (T = npos C G) whole, in K order, so a piece's value
does not depend on T (batch invariance). Weights of a layer: one stack of NE + 1
experts, expert e's units at e upe .. in the dense unit order (pack.unit_order).

  "gu"   (S5): x = norm(h1) (all K in LDS). Partial of piece -> part[pi]; counter per
         (pos, g) counts C; the last arriver sums the chunks in order and stores act =
         silu(gate) * up for the rows that use pos: act[m][slot][8 g + col].
  "down" (S6): x = act of the CTA's <= 2 positions (prologue_act_moe). Partial of
         piece (pos, c, g) -> part[g][slot][c][col][m] for the rows that use pos;
         counter per g counts npos C; the last arriver forms, per row, sum over j in
         order of w[m][j] * (sum over c of the slot's partials) and runs the residual
         epilogue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import stages as S
from .dims import NE, TOPK
from .mkcore import dev as D
from .mkcore.gemv import fp8_b128_compute, fp8_b128_load

LGN = 272  # router GEMV rows: 256 experts, the shared gate, 15 zero rows
NSLOT = TOPK + 1  # 8 routed slots + the shared expert
NEG = -3.0e38
NSUB = 4  # S6 sub-counters per group (two-level counting, MoeGemv._cg)


def i2f(i):
    """Int 0 <= i < 2^23 -> the same value as f32 (exact; no conversion op)."""
    return (fx.Int32(i) | 0x4B000000).bitcast(fx.Float32) - fx.Float32(8388608.0)


def f2i(f):
    """Inverse of i2f for integral f in [0, 2^23)."""
    return (fx.Float32(f) + fx.Float32(8388608.0)).bitcast(fx.Int32) & 0x7FFFFF


def npos_max(M: int) -> int:
    return min(TOPK * M, NE) + 1


class Route:
    """LDS pointers of the routing tables (allocated by the kernel)."""

    def __init__(self, M, pick, w, eid, posof, slot, npos, rmask):
        self.M, self.pick, self.w, self.eid = M, pick, w, eid
        self.posof, self.slot, self.npos, self.rmask = posof, slot, npos, rmask
        self.NP = npos_max(M)

    def n(self):
        return D.uniform(fx.ptr_load(self.npos))

    def eid_of(self, p):
        return D.uniform(fx.ptr_load(self.eid + p))


def slot_of(slot_ptr, NP, m, p):
    """Row m's slot for position p, or -1 (a plain function: FlyDSL carries the base
    object of a method call through control flow, and Route is not an IR value)."""
    return fx.ptr_load(slot_ptr + (m * NP + p))


@D.traced
def routing(c, M, logits, rowv, R: Route):
    """Build R's tables from logits fp32 [M][LGN] (device scope) and the row-valid
    flags rowv i32 [M]. Wave m does row m; wave 0 builds the position list."""
    lane, wave, tid = c["lane"], c["wave"], c["tid"]
    NT = S.NT
    D.PROBE["on"] = D.PROBE.get("stage") == "route"
    D.probe(c, 9)
    if wave < M:
        rl = D.rsrc(logits)
        m = wave
        valid = D.uniform(D.ld(D.rsrc(rowv), m)) != 0
        v = [D.ldf(rl, m * LGN + lane + 64 * k, cm=D.CM_DEV) for k in range(4)]
        sg = D.ldf(rl, m * LGN + NE, cm=D.CM_DEV)
        vals, picks = [], []
        for j in range_constexpr(TOPK):
            if const_expr(j == 1):
                D.probe(c, 10)  # logits loaded, first max done
            best, bk = v[0], fx.Int32(0)
            for k in range_constexpr(1, 4):
                gt = v[k] > best  # strict: the lower index wins a tie
                best = gt.select(v[k], best)
                bk = gt.select(fx.Int32(k), bk)
            # wave-uniform maxima by DPP only (dev.wave_max); max is exact in any order
            mv = D.wave_max(best)
            idx = lane + 64 * bk
            cand = (best == mv).select(-i2f(idx), fx.Float32(NEG))
            e = D.uniform(f2i(-D.wave_max(cand)))
            vals.append(mv)
            picks.append(e)
            for k in range_constexpr(4):
                v[k] = ((lane + 64 * k) == e).select(fx.Float32(NEG), v[k])
        ws = [D.expf(x - vals[0]) for x in vals]
        tot = ws[0]
        for j in range_constexpr(1, TOPK):
            tot = tot + ws[j]
        D.probe(c, 11)  # top-8 done
        if lane == 0:
            for j in range_constexpr(TOPK):
                fx.ptr_store(ws[j] / tot, R.w + (m * NSLOT + j))
                fx.ptr_store(
                    valid.select(picks[j], fx.Int32(-1)), R.pick + (m * TOPK + j)
                )
            fx.ptr_store(D.sigmoid(sg), R.w + (m * NSLOT + TOPK))
        # row m's expert set as 8 words of 32 bits (lane l < 8 builds word l from the
        # wave-uniform picks; no memory round trip); a padding row selects none
        word = fx.Int32(0)
        for j in range_constexpr(TOPK):
            hit = valid & ((picks[j] >> 5) == lane)
            word = word | hit.select(fx.Int32(1) << (picks[j] & 31), fx.Int32(0))
        if lane < 8:
            fx.ptr_store(word, R.rmask + (m * 8 + lane))
    gpu.barrier()
    D.probe(c, 12)
    if wave == 0:
        # lane l owns experts 4 l .. 4 l + 3: bits 4 (l % 8) .. + 3 of word l // 8
        wd = fx.Int32(0)
        for m in range_constexpr(M):
            wd = wd | fx.ptr_load(R.rmask + (m * 8 + lane // 8))
        nib = (wd >> (4 * (lane % 8))) & 15
        bits = [(nib >> b) & 1 for b in range(4)]
        cnt = bits[0] + bits[1] + bits[2] + bits[3]
        inc = cnt
        for off in (1, 2, 4, 8, 16, 32):
            o = D.gather(inc, fx.max(lane - off, 0))
            inc = inc + (lane >= off).select(o, fx.Int32(0))
        total = D.uniform(D.gather(inc, 63))
        p = inc - cnt
        for b in range_constexpr(4):
            sel = bits[b] != 0
            if sel:
                fx.ptr_store(4 * lane + b, R.eid + p)
            fx.ptr_store(sel.select(p, fx.Int32(-1)), R.posof + (4 * lane + b))
            p = p + bits[b]
        if lane == 0:
            fx.ptr_store(fx.Int32(NE), R.eid + total)
            fx.ptr_store(total + 1, R.npos)
    n = M * R.NP
    for i in range_constexpr((n + NT - 1) // NT):
        k = tid + NT * i
        if k < n:
            fx.ptr_store(fx.Int32(-1), R.slot + k)
    gpu.barrier()
    D.probe(c, 13)
    npos = fx.ptr_load(R.npos)
    if tid < M * TOPK:
        e = fx.ptr_load(R.pick + tid)
        if e >= 0:
            m = tid // TOPK
            fx.ptr_store(tid % TOPK, R.slot + (m * R.NP + fx.ptr_load(R.posof + e)))
    if tid < M:
        fx.ptr_store(fx.Int32(TOPK), R.slot + (tid * R.NP + npos - 1))
    gpu.barrier()
    D.probe(c, 14)


@D.traced
def route_hash(c, M, R: Route, out, li, cus, nl):
    """Debug (invariant I5): thread 0 of each CTA stores a hash of its routing tables
    (npos, eid, slot, pick, w bits) to the i32 out[li][bid]; CTA 0 also stores its pick
    table to out[nl cus + (li M + m) 8 + j] and its weights (f32 bits) to
    out[nl cus + nl M 8 + (li M + m) 9 + j]."""
    eid, slot, pick, wt = R.eid, R.slot, R.pick, R.w
    tid, bid = c["tid"], c["bid"]
    ro = D.rsrc(out)
    if tid == 0:
        npos = fx.ptr_load(R.npos)
        h = npos
        for k in range(0, npos):
            h = h * 31 + fx.ptr_load(eid + k)
        for k in range(0, M * R.NP):
            h = h * 31 + fx.ptr_load(slot + k)
        for k in range(0, M * TOPK):
            h = h * 31 + fx.ptr_load(pick + k)
        for k in range(0, M * NSLOT):
            h = h * 31 + fx.ptr_load(wt + k).bitcast(fx.Int32)
        D.st(h, ro, li * cus + bid, D.CM_DEV)
    if bid == 0:
        if tid < M * TOPK:
            D.st(fx.ptr_load(pick + tid), ro, nl * cus + li * M * TOPK + tid, D.CM_DEV)
        if tid < M * NSLOT:
            D.st(
                fx.ptr_load(wt + tid).bitcast(fx.Int32),
                ro,
                nl * cus + nl * M * TOPK + li * M * NSLOT + tid,
                D.CM_DEV,
            )


@dataclass(frozen=True)
class MoeGemv:
    """Grouped fp8 GEMV of one MoE stage (see the module docstring). N, K per expert."""

    name: str
    N: int
    K: int
    M: int
    cus: int
    mode: str  # "gu" | "down"
    waves: int = 8
    C: int = 1  # K chunks per expert

    kind = "fp8_b128"
    esize = 1
    slots = 1  # (prologue_norm interface: all K in LDS)

    @property
    def G(self):
        return self.N // 16

    @property
    def KS(self):
        return self.K // 128

    @property
    def KSp(self):
        return self.KS // self.C

    @property
    def Kc(self):
        """Columns of x per LDS row (prologue interface): all of K."""
        return self.K

    @property
    def ppe(self):
        return self.C * self.G

    @property
    def upe(self):
        return self.G * self.KS

    @property
    def NP(self):
        return npos_max(self.M)

    @property
    def NW(self):
        return self.cus * self.waves

    @property
    def PMAX(self):
        """Max pieces of one wave."""
        return math.ceil(self.NP * self.ppe / self.NW)

    @property
    def XROWS(self):
        """LDS x rows: M tokens (gu), or 16 per position slot x 2 slots (down)."""
        return self.M if self.mode == "gu" else 32

    @property
    def XR(self):
        return self.K + 16

    @property
    def XSR(self):
        return self.KS

    @property
    def MT(self):
        return (self.M + 3) // 4

    def chunk0(self, bid):
        return fx.Int32(0)

    def check(self):
        assert self.N % 16 == 0 and self.K % 128 == 0 and self.KS % self.C == 0
        assert 1 <= self.KSp <= 4
        if self.mode == "down":
            # I6: a CTA's range touches <= 2 positions
            assert math.ceil(self.NP * self.ppe / self.cus) <= self.ppe, self
        assert (self.NE1 * self.upe + 1) * 2048 < 2**31
        return self

    NE1 = NE + 1

    def part_floats(self):
        if self.mode == "gu":
            return self.NP * self.ppe * 256
        return self.G * NSLOT * self.C * 256

    # ------------------------------------------------------------ device
    def setup(self, c, xq, xs, tiles, scales):
        lane, gl, col = c["lane"], c["gl"], c["col"]
        return dict(
            lane=lane,
            gl=gl,
            col=col,
            xq=xq,
            xs=xs,
            rt=D.rsrc(tiles),
            rs=D.rsrc(scales),
            gw=c["bid"] * self.waves + c["wave"],
            arow=fx.min(col, self.M - 1) * self.XR + 8 * gl,
        )

    def cta_pos0(self, bid, npos):
        """The first position of CTA bid's piece range (S6 prologue)."""
        T_ = npos * self.ppe
        return ((bid * self.waves) * T_ // self.NW) // self.ppe

    def _ubase(self, R, pi):
        pos = pi // self.ppe
        return R.eid_of(pos) * self.upe + (pi % self.ppe) * self.KSp

    def _load_piece(self, w, R, pi):
        ub = self._ubase(R, pi)
        return [
            fp8_b128_load(w["rt"], w["rs"], ub + s, w["lane"]) for s in range(self.KSp)
        ]

    @D.traced
    def run(self, c, w, R, part, cnt, fin, pos0=None):
        """Stream this wave's pieces, then count them; the last arriver of a counter
        group calls fin(cg, tots[, ...]) (see mode). pos0: the CTA's first position
        (down mode: LDS x slot = pos - pos0)."""
        D.PROBE["on"] = D.PROBE.get("stage") == self.name
        D.probe(c, 0)
        npos = R.n()
        T_ = npos * self.ppe
        gw = w["gw"]
        p0 = gw * T_ // self.NW
        p1 = (gw + 1) * T_ // self.NW
        rp = D.rsrc(part)
        G, KSp = self.G, self.KSp
        rslot = R.slot
        buf = self._load_piece(w, R, fx.min(p0, T_ - 1))
        for pi in range(p0, p1):
            nxt = self._load_piece(w, R, fx.min(pi + 1, T_ - 1))
            pos = pi // self.ppe
            pl = pi % self.ppe
            ch = pl // G
            g = pl % G
            arow = w["arow"]
            xof = fx.Int32(0)
            if const_expr(self.mode == "down"):
                sl = pos - pos0
                arow = arow + sl * 16 * self.XR
                xof = sl * 16 * self.XSR
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for s in range_constexpr(KSp):
                xo = ch * KSp + s
                acc = acc + fp8_b128_compute(
                    w, arow + xo * 128, xof + xo, self.XSR, buf[s]
                )
            if const_expr(self.mode == "gu"):
                if w["gl"] < self.MT:
                    D.st(acc, rp, (pi * 64 + w["lane"]) * 4, D.CM_DEV)
            else:
                for r in range_constexpr(4):
                    m = 4 * w["gl"] + r
                    if m < self.M:
                        j = slot_of(rslot, self.NP, m, pos)
                        if j >= 0:
                            D.st(acc[r], rp, self._p2(g, j, ch, w["col"], m), D.CM_DEV)
            buf = nxt
        D.probe(c, 1)
        rocdl.s_waitcnt(vmcnt=0)
        olds = []
        for k in range_constexpr(self.PMAX):
            pi = p0 + k
            old = fx.Int32(0)
            if (w["lane"] == 0) & (pi < p1):
                old = D.atomic_add_dev(cnt + self._cg(pi) * 4, 1)
            olds.append(D.uniform(old))
        D.probe(c, 3)
        for k in range_constexpr(self.PMAX):
            pi = p0 + k
            cg = self._cg(pi)
            if const_expr(self.mode == "gu"):
                total = fx.Int32(self.C)
            else:
                sub = pi // self.ppe % NSUB
                total = self.C * ((npos - sub + NSUB - 1) // NSUB)
            if (pi < p1) & ((olds[k] + 1) == total):
                if w["lane"] == 0:
                    D.st(fx.Int32(0), D.rsrc(cnt), cg, D.CM_DEV)
                if const_expr(self.mode == "gu"):
                    self._reduce_gu(w, R, rp, pi // self.ppe, pi % G, fin)
                else:
                    # level 2: the last of the min(npos, NSUB) sub-counters reduces
                    gg = pi % G
                    old2 = fx.Int32(0)
                    if w["lane"] == 0:
                        old2 = D.atomic_add_dev(cnt + (NSUB * G + gg) * 4, 1)
                    if (D.uniform(old2) + 1) == fx.min(npos, NSUB):
                        if w["lane"] == 0:
                            D.st(fx.Int32(0), D.rsrc(cnt), NSUB * G + gg, D.CM_DEV)
                        self._reduce_down(w, R, rp, gg, fin)
        D.probe(c, 2)

    def _cg(self, pi):
        """Counter of piece pi. gu: (position, group), C arrivals. down: (pos % NSUB,
        group), C arrivals per position of the sub-set; the sub-counters' last arrivers
        meet at counter NSUB G + group (spreads the atomics of one group over NSUB
        addresses: measured contention at M = 8)."""
        if self.mode == "gu":
            return (pi // self.ppe) * self.G + pi % self.G
        return (pi // self.ppe % NSUB) * self.G + pi % self.G

    def _p2(self, g, j, ch, col, m):
        return (((g * NSLOT + j) * self.C + ch) * 16 + col) * 16 + m

    @D.traced
    def _reduce_gu(self, w, R, rp, pos, g, fin):
        vs = [
            D.ld(
                rp,
                (((pos * self.ppe + ch * self.G + g) * 64) + w["lane"]) * 4,
                4,
                T.f32,
                D.CM_DEV,
            )
            for ch in range(self.C)
        ]
        tot = vs[0]
        for ch in range_constexpr(1, self.C):
            tot = tot + vs[ch]
        fin(pos, g, tot)

    @D.traced
    def _reduce_down(self, w, R, rp, g, fin):
        pre = fin.pre(g)
        vs = []
        for j in range_constexpr(NSLOT):
            vs.append(
                [
                    D.ld(
                        rp,
                        self._p2(g, j, ch, w["col"], 4 * w["gl"]),
                        4,
                        T.f32,
                        D.CM_DEV,
                    )
                    for ch in range(self.C)
                ]
            )
        tot = []
        for r in range_constexpr(4):
            m = fx.min(4 * w["gl"] + r, self.M - 1)
            t = fx.Float32(0.0)
            for j in range_constexpr(NSLOT):
                s = vs[j][0][r]
                for ch in range_constexpr(1, self.C):
                    s = s + vs[j][ch][r]
                wj = fx.ptr_load(R.w + (m * NSLOT + j))
                if const_expr(j < TOPK):
                    used = fx.ptr_load(R.pick + (m * TOPK + j)) >= 0
                    wj = used.select(wj, fx.Float32(0.0))
                    s = used.select(s, fx.Float32(0.0))
                t = t + wj * s
            tot.append(t)
        fin(g, [fx.Vector.from_elements(tot, fx.Float32)], pre)


class EpiAct:
    """S5 last arriver: group g of position pos holds gate rows 8 g .. + 7 (lanes col <
    8) and the up rows (col + 8) of act columns 8 g + col; store silu(gate) * up to
    act[m][slot][.] fp32 [M][NSLOT][FI] for the rows that use pos."""

    def __init__(self, c, M, R, act, fi):
        self.c, self.M, self.R, self.act, self.fi = c, M, R, act, fi

    def __call__(self, pos, g, tot):
        _epi_act(self.c, self.M, self.R, self.act, self.fi, pos, g, tot)


@D.traced
def _epi_act(c, M, R, act, fi, pos, g, tot):
    col, lane = c["col"], c["lane"]
    ra = D.rsrc(act)
    rslot = R.slot
    for r in range_constexpr(4):
        m = 4 * c["gl"] + r
        up = D.gather_f(tot[r], lane + 8)
        if (m < M) & (col < 8):
            j = slot_of(rslot, R.NP, m, pos)
            if j >= 0:
                D.st(
                    D.silu(tot[r]) * up,
                    ra,
                    (m * NSLOT + j) * fi + g * 8 + col,
                    D.CM_DEV,
                )


@D.traced
def prologue_act_moe(c, g, xq, xs, act, R, pos0):
    """S6 LDS x <- act rows of positions pos0, pos0 + 1 (slot rows sl * 16 + m),
    quantized per (row, 128 K). Rows that do not use a position read slot 0 (their
    results are never stored); rows m >= M get a zero scale."""
    NT = S.NT

    M, K = g.M, g.K
    tid = c["tid"]
    npos = R.n()
    nz = 32 * g.XSR
    for i in range_constexpr((nz + NT - 1) // NT):
        k = tid + NT * i
        if k < nz:
            fx.ptr_store(fx.Float32(0.0), xs + k)
    gpu.barrier()
    ra = D.rsrc(act)
    per = K // 8
    nch = 2 * M * per
    raw = []
    for i in range_constexpr((nch + NT - 1) // NT):
        k = fx.min(tid + NT * i, nch - 1)
        sl = k // (M * per)
        m = k % (M * per) // per
        kc = k % per * 8
        pos = fx.min(pos0 + sl, npos - 1)
        j = fx.max(slot_of(R.slot, R.NP, m, pos), 0)
        base = (m * NSLOT + j) * K + kc
        raw.append(
            (i, sl, m, kc, [D.ld(ra, base + o, 4, T.f32, D.CM_DEV) for o in (0, 4)])
        )
    for i, sl, m, kc, v in raw:
        x = [v[0][0], v[0][1], v[0][2], v[0][3], v[1][0], v[1][1], v[1][2], v[1][3]]
        S.store_quant8_at(
            c, xq, xs, x, sl * 16 + m, g.XR, g.XSR, kc, tid + NT * i < nch
        )
    gpu.barrier()
