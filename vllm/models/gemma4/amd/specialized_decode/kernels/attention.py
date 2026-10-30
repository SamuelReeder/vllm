# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Gemma 4 decode attention over vLLM's paged KV cache (FlyDSL wave tasks).

Runs between two grid barriers. Input: the raw fp32 output of the qkv GEMV of one layer,
y[M][QKV_N] = [q | k | v] (full layers: [q | k], V is the k projection: attention_k_eq_v).
Output: o[M][NQ HD] fp32, the o_proj input.

Task = (row m, kv head j, q-head group g, chunk slot r), one wave. Chunk ci = c_lo + r covers
positions [64 ci, 64 ci + 64); c_lo is the first chunk that holds a visible position:
  sliding layers: positions lo..pos, lo = max(0, pos - (WINDOW - 1)), at most 17 chunks;
  full layers: positions 0..pos.
Steps of a task:
  q (GQT heads), k_new: RMSNorm over HD with weight (x * w), NeoX RoPE (pairs d, d + HD/2;
  the RoPE table holds cos = 1, sin = 0 for the identity pairs of the proportional RoPE);
  v_new: RMSNorm without weight, no RoPE.
  Scores on the bf16 MFMA (q in bf16), scale 1.0, softmax partials, P V on the VALU.
  The task whose chunk holds pos (group 0) writes k_new, v_new (bf16) at slot_mapping[m].
  Partial record [m][j][g][r]: max[GQT], sum[GQT] (padded to 16 floats), o[GQT][HD].
  The last arriver of (m, j, g) merges the records, writes o and resets the counter.
Paged KV: vLLM TRITON_ATTN, position p of row m is in block bt[m][p // B] at offset p % B;
element strides (block s0, head s1, position s2), V sv elements after K. B % 16 == 0, so a
16-position score tile or P V batch lies in one block: one block-table lookup each.
Every position a task loads is clamped to [max(lo, p0), min(pos, p0 + 63)]: no task reads
a block-table entry outside the window (vLLM nulls them; the test fills them with NaN).
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from .dims import EPS, HD_F, HD_S, NKV_F, NKV_S, NQ, QKV_F, QKV_S, WINDOW
from .mkcore import dev as D

CH = 64  # positions per task
PVB = 16  # P V: positions per load batch (one block)
NEG = -3.0e38


@dataclass(frozen=True)
class Geo:
    """One layer type's attention geometry and its cache layout (compile-time; with
    kernel.Cfg.rt_geom, NC of full layers and maxblocks are run-time i32 values)."""

    full: bool
    NC: int  # chunk slots per (row, kv head, group)
    maxblocks: int  # block-table columns
    block: int  # KV positions per block (B)
    s0: int  # element strides: block, head, position
    s1: int
    s2: int
    sv: int  # elements from a K row to its V row

    @property
    def HD(self):
        return HD_F if self.full else HD_S

    @property
    def NKV(self):
        return NKV_F if self.full else NKV_S

    @property
    def NG(self):
        """q-head groups per kv head: full layers split their 8 heads in 2 (LDS)."""
        return 2 if self.full else 1

    @property
    def GQT(self):
        return NQ // self.NKV // self.NG

    @property
    def QKV_N(self):
        return QKV_F if self.full else QKV_S

    @property
    def K_OFF(self):
        return NQ * self.HD

    @property
    def V_OFF(self):
        return self.K_OFF if self.full else self.K_OFF + self.NKV * self.HD

    @property
    def DL(self):
        return self.HD // 64  # dims per lane

    @property
    def NU(self):
        return self.HD // 32  # 16 B K loads per lane per position

    @property
    def REC(self):
        return 16 + self.GQT * self.HD

    @property
    def WLF(self):
        """LDS floats per wave: q (bf16 pairs; then scores / probabilities), k_new, v_new."""
        return self.GQT * self.HD // 2 + self.HD

    def ntask(self, M):
        return M * self.NKV * self.NG * self.NC

    def nrec(self, M):
        return self.ntask(M) * self.REC

    def ncnt(self, M):
        return M * self.NKV * self.NG

    def check(self):
        assert self.block % 16 == 0, "a 16-position tile must lie in one KV block"
        assert self.GQT * CH <= self.GQT * self.HD // 2
        assert self.GQT <= 8 and self.DL in (4, 8)
        if not self.full:
            assert self.NC >= WINDOW // CH + 1
        return self


def _ld_n(r, i, n, dtype=None, cm=0):
    """N (2, 4 or 8) elements from i as a list (8 = two loads of 4)."""
    if n <= 4:
        v = D.ld(r, i, n, dtype, cm)
        return [v[e] for e in range(n)]
    a, b = D.ld(r, i, 4, dtype, cm), D.ld(r, i + 4, 4, dtype, cm)
    return [a[e] for e in range(4)] + [b[e] for e in range(4)]


def _bf_words(ws):
    out = []
    for w in ws:
        out += [D.bf_lo(w), D.bf_hi(w)]
    return out


def _pack_bf2(a, b):
    """q, K, V to bf16 with round to nearest even (dev.pack_bf2), as torch and vLLM round:
    the cache holds the bits vLLM's own path would write."""
    return D.pack_bf2_rne(a, b)


def _lds_st_i32(base_f32, idx, words):
    """Store i32 words (2 or 4) at float index idx of an f32 LDS pointer."""
    for k in range_constexpr(0, len(words), 4):
        part = words[k : k + 4]
        fx.ptr_store(
            fx.Vector.from_elements(part, fx.Int32).bitcast(fx.Float32),
            base_f32 + (idx + k),
        )


def _lds_ld_i32(base_f32, idx, n):
    v = fx.Vector(
        fx.ptr_load(base_f32 + idx, result_type=fx.Vector.make_type(n, fx.Float32))
    ).bitcast(fx.Int32)
    return [v[e] for e in range(n)]


def norm_rope(geo, lane, vals, wvals, cos, sin, rope):
    """RMSNorm over HD (DL dims a lane, dims DL lane + e) times w (None: no weight), then
    NeoX RoPE: pair (d, d + HD/2) lives in lanes (l, l ^ 32)."""
    ss = vals[0] * vals[0]
    for e in range_constexpr(1, geo.DL):
        ss = ss + vals[e] * vals[e]
    ss = D.xor_sum(ss, lane, (1, 2, 4, 8, 16, 32))
    rn = D.rsq(ss / fx.Float32(float(geo.HD)) + fx.Float32(EPS))
    x = [
        vals[e] * rn if wvals is None else vals[e] * rn * wvals[e]
        for e in range(geo.DL)
    ]
    if not rope:
        return x
    lo = lane < 32
    out = []
    for e in range_constexpr(geo.DL):
        partner = D.gather_f(x[e], lane ^ 32)
        out.append(
            lo.select(
                x[e] * cos[e] - partner * sin[e], x[e] * cos[e] + partner * sin[e]
            )
        )
    return out


@D.traced
def attn_task(
    c, geo, task, y, qn_w, kn_w, rope_cs, kv, bt, slotp, pos_ptr, rec, cnt, o, wl
):
    """One (m, j, g, r) task. rope_cs: [maxlen][HD] fp32 = cos[HD/2] | sin[HD/2] per pair.
    wl: this wave's LDS (geo.WLF floats)."""
    per_m = geo.NKV * geo.NG * geo.NC
    m = task // per_m
    rem = task % per_m
    j = rem // (geo.NG * geo.NC)
    g = rem % (geo.NG * geo.NC) // geo.NC
    r = rem % geo.NC
    pos = D.uniform(D.ld(D.rsrc(pos_ptr), m))
    slot = D.uniform(D.ld(D.rsrc(slotp), m))
    lo = fx.Int32(0)
    if const_expr(not geo.full):
        lo = fx.max(pos - (WINDOW - 1), fx.Int32(0))
    c_lo = lo // CH
    p0 = (c_lo + r) * CH
    if (p0 <= pos) & (slot >= 0):
        _chunk(
            c,
            geo,
            m,
            j,
            g,
            r,
            pos,
            slot,
            lo,
            c_lo,
            p0,
            y,
            qn_w,
            kn_w,
            rope_cs,
            kv,
            bt,
            rec,
            cnt,
            o,
            wl,
        )


@D.traced
def _chunk(
    c,
    geo,
    m,
    j,
    g,
    r,
    pos,
    slot,
    lo,
    c_lo,
    p0,
    y,
    qn_w,
    kn_w,
    rope_cs,
    kv,
    bt,
    rec,
    cnt,
    o,
    wl,
):
    lane = c["lane"]
    D.PROBE["on"] = D.PROBE.get("stage") == "attn"  # bench/profile_decode.py PROBE=attn
    D.probe(c, 4)
    HD, DL, NU, GQT, B = geo.HD, geo.DL, geo.NU, geo.GQT, geo.block
    s2, sv = geo.s2, geo.sv
    d0 = DL * lane
    first = fx.max(lo, p0)
    lastp = fx.min(pos, p0 + (CH - 1))
    p = p0 + lane
    live = (p >= first) & (p <= lastp)
    gq, nq = lane // 16, lane % 16  # MFMA lane coordinates
    rbt = D.rsrc(bt)
    QB, KN, VN = 0, GQT * HD // 2, GQT * HD // 2 + HD // 2  # LDS float offsets

    def clamp(v):
        return fx.min(fx.max(v, first), lastp)

    def block_rsrc(pp):
        """(buffer resource at (block of pp, head j), block's first position); pp uniform."""
        bi = pp // B
        blk = D.uniform(D.ld(rbt, m * geo.maxblocks + bi))
        return D.rsrc(kv + (fx.Int64(blk) * geo.s0 + j * geo.s1) * 2), bi * B

    def ktile(t):
        """B operand of score tile t: position p0 + 16 t + nq (clamped). Load u of lane group
        gq: dims 32 u + 8 gq .. + 7; k-step 2 u + h = dims 32 u + 8 gq + 4 h .. + 3."""
        rk, pb = block_rsrc(clamp(p0 + 16 * t))
        pt = clamp(p0 + 16 * t + nq)
        return [D.ld(rk, ((pt - pb) * s2 + 32 * u + 8 * gq) // 2, 4) for u in range(NU)]

    def vload(b):
        """[k] -> DL/2 bf16-pair words of dims d0.. of position p0 + PVB b + k (clamped)."""
        rk, pb = block_rsrc(clamp(p0 + PVB * b))
        out = []
        for k in range_constexpr(PVB):
            pk = clamp(p0 + PVB * b + k)
            v = D.ld(rk, ((pk - pb) * s2 + sv + d0) // 2, DL // 2)
            out.append([v[e] for e in range(DL // 2)])
        return out

    kt = ktile(0)  # independent of this step's q / k / v
    # ---- q, k_new, v_new
    ry = D.rsrc(y)
    ybase = m * geo.QKV_N
    rcs = D.rsrc(rope_cs)
    pi = d0 % (HD // 2)
    cos = _ld_n(rcs, pos * HD + pi, DL, T.f32)
    sin = _ld_n(rcs, pos * HD + HD // 2 + pi, DL, T.f32)
    qw = _bf_words(_ld_n(D.rsrc(qn_w), d0 // 2, DL // 2))
    kw = _bf_words(_ld_n(D.rsrc(kn_w), d0 // 2, DL // 2))
    for hh in range_constexpr(GQT):
        h = j * (NQ // geo.NKV) + g * GQT + hh
        qv = _ld_n(ry, ybase + h * HD + d0, DL, T.f32, D.CM_DEV)
        q = norm_rope(geo, lane, qv, qw, cos, sin, True)
        _lds_st_i32(
            wl,
            QB + hh * HD // 2 + d0 // 2,
            [_pack_bf2(q[2 * e], q[2 * e + 1]) for e in range(DL // 2)],
        )
    kr = _ld_n(ry, ybase + geo.K_OFF + j * HD + d0, DL, T.f32, D.CM_DEV)
    vr = kr
    if const_expr(not geo.full):
        vr = _ld_n(ry, ybase + geo.V_OFF + j * HD + d0, DL, T.f32, D.CM_DEV)
    kn = norm_rope(geo, lane, kr, kw, cos, sin, True)
    vn = norm_rope(geo, lane, vr, None, cos, sin, False)
    kw2 = [_pack_bf2(kn[2 * e], kn[2 * e + 1]) for e in range(DL // 2)]
    vw2 = [_pack_bf2(vn[2 * e], vn[2 * e + 1]) for e in range(DL // 2)]
    _lds_st_i32(wl, KN + d0 // 2, kw2)
    _lds_st_i32(wl, VN + d0 // 2, vw2)
    if (g == 0) & (pos >= p0) & (pos < p0 + CH):
        # the new position's K / V at its slot (vLLM's slot_mapping: block * B + offset)
        rs = D.rsrc(kv + (fx.Int64(slot // B) * geo.s0 + j * geo.s1) * 2)
        so = slot % B
        D.st(fx.Vector.from_elements(kw2, fx.Int32), rs, (so * s2 + d0) // 2, D.CM_DEV)
        D.st(
            fx.Vector.from_elements(vw2, fx.Int32),
            rs,
            (so * s2 + sv + d0) // 2,
            D.CM_DEV,
        )
    D.probe(c, 5)
    # ---- scores on the MFMA (bf16 16x16x16): tile [16 head rows][16 positions]; rows >= GQT
    # are zero. q goes to registers first: its LDS region then takes the scores.
    qa = []
    z2 = [fx.Int32(0), fx.Int32(0)]
    for i in range_constexpr(2 * NU):
        w2 = _lds_ld_i32(
            wl,
            QB
            + fx.min(nq, GQT - 1) * (HD // 2)
            + (32 * (i // 2) + 8 * gq + 4 * (i % 2)) // 2,
            2,
        )
        qa.append([(nq < GQT).select(w2[e], z2[e]) for e in range(2)])
    for t in range(0, D.opaque_s(CH // 16)):  # opaque trip count: LLVM would unroll
        kn_ = ktile(fx.min(t + 1, CH // 16 - 1))
        _score_tile(geo, wl, t, p0, pos, gq, nq, qa, kt, KN)
        kt = kn_
    D.probe(c, 6)
    mx, sm = [], []
    for hh in range_constexpr(GQT):
        s_ = live.select(fx.ptr_load(wl + (hh * CH + lane)), fx.Float32(NEG))
        mh = D.xor_max(s_, lane, (1, 2, 4, 8, 16, 32))
        ph = live.select(D.expf(s_ - mh), fx.Float32(0.0))
        mx.append(mh)
        sm.append(D.xor_sum(ph, lane, (1, 2, 4, 8, 16, 32)))
        fx.ptr_store(ph, wl + (hh * CH + lane))
    # ---- P V: lane -> dims d0 .. d0 + DL - 1. Loading batch b + 1 during batch b was
    # slower (bench/results/m4/attn_vpipe*.txt; cause not measured, likely VGPR pressure)
    acc = [[fx.Float32(0.0)] * DL for _ in range(GQT)]
    vl_ = _lds_ld_i32(wl, VN + d0 // 2, DL // 2)
    for b in range(0, D.opaque_s(CH // PVB)):
        vrs = vload(b)
        acc = _pv_batch(geo, wl, b, p0, pos, vl_, vrs, acc)
    D.probe(c, 7)
    # ---- partial record
    rr = D.rsrc(rec)
    cidx = (m * geo.NKV + j) * geo.NG + g
    rbase = (cidx * geo.NC + r) * geo.REC
    if lane < GQT:
        D.st(_sel(lane, mx), rr, rbase + lane, D.CM_DEV)
        D.st(_sel(lane, sm), rr, rbase + 8 + lane, D.CM_DEV)
    for hh in range_constexpr(GQT):
        for e0 in range_constexpr(0, DL, 4):
            D.st(
                fx.Vector.from_elements(acc[hh][e0 : e0 + 4], fx.Float32),
                rr,
                rbase + 16 + hh * HD + d0 + e0,
                D.CM_DEV,
            )
    # ---- last arriver of (m, j, g) merges
    rocdl.s_waitcnt(vmcnt=0)
    nchunk = pos // CH - c_lo + 1
    old = fx.Int32(0)
    if lane == 0:
        old = D.atomic_add_dev(cnt + cidx * 4, 1)
    old = D.uniform(old)
    D.probe(c, 8)
    if old + 1 == nchunk:
        # every other arrival is in; the counter's next use is behind a grid barrier
        if lane == 0:
            D.st(fx.Int32(0), D.rsrc(cnt), cidx, D.CM_DEV)
        _merge(geo, lane, m, j, g, cidx, nchunk, rr, o)
        D.probe(c, 9)


@D.traced
def _score_tile(geo, wl, t, p0, pos, gq, nq, qa, kt, KN):
    """Scores of positions p0 + 16 t .. + 15 for the GQT heads -> LDS s[h][16 t + n]. The
    new position's k comes from LDS (bf16 pairs), the rest from the cache."""
    isnew = (p0 + 16 * t + nq) == pos
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for u in range_constexpr(geo.NU):
        kl = _lds_ld_i32(wl, KN + (32 * u + 8 * gq) // 2, 4)
        kw = [isnew.select(kl[e], kt[u][e]) for e in range(4)]
        for h in range_constexpr(2):
            a = fx.Vector.from_elements(qa[2 * u + h], fx.Int32).bitcast(fx.Int16)
            b = fx.Vector.from_elements([kw[2 * h], kw[2 * h + 1]], fx.Int32).bitcast(
                fx.Int16
            )
            acc = fx.Vector(
                rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [a, b, acc, 0, 0, 0])
            )
    # D[row 4 gq + r][col nq]: head 4 gq + r, position 16 t + nq
    for r in range_constexpr(4):
        if 4 * gq + r < geo.GQT:
            fx.ptr_store(acc[r], wl + ((4 * gq + r) * CH + 16 * t + nq))


@D.traced
def _pv_batch(geo, wl, b, p0, pos, vl_, vrs, acc):
    """Acc += p V over positions p0 + PVB b + k (p = 0 outside the visible range)."""
    acc = [list(a) for a in acc]
    for k in range_constexpr(PVB):
        pp = b * PVB + k
        nw = (p0 + pp) == pos
        ws = [nw.select(vl_[e], vrs[k][e]) for e in range(geo.DL // 2)]
        vf = _bf_words(ws)
        for hh in range_constexpr(geo.GQT):
            ph = fx.ptr_load(wl + (hh * CH + pp))
            for e in range_constexpr(geo.DL):
                acc[hh][e] = D.fma(ph, vf[e], acc[hh][e])
    return acc


def _sel(lane, vals):
    v = vals[0]
    for i in range(1, len(vals)):
        v = (lane == i).select(vals[i], v)
    return v


MERGE_B = 2  # records per merge batch


@D.traced
def _merge(geo, lane, m, j, g, cidx, nchunk, rr, o):
    """Online-softmax merge of the (m, j, g) records -> o[m][h HD + d] (fp32)."""
    HD, DL, GQT = geo.HD, geo.DL, geo.GQT
    d0 = DL * lane
    base = cidx * geo.NC * geo.REC
    gmax = [fx.Float32(NEG)] * GQT
    tot = [fx.Float32(0.0)] * GQT
    acc = [[fx.Float32(0.0)] * DL for _ in range(GQT)]
    for b0 in range(0, (nchunk + MERGE_B - 1) // MERGE_B):
        recs = []
        for k in range_constexpr(MERGE_B):
            ci = fx.min(b0 * MERGE_B + k, nchunk - 1)
            ok = (b0 * MERGE_B + k) < nchunk
            rb = base + ci * geo.REC
            ms = _ld_n(rr, rb, 8, T.f32, D.CM_DEV)
            ls = _ld_n(rr, rb + 8, 8, T.f32, D.CM_DEV)
            os_ = [
                _ld_n(rr, rb + 16 + hh * HD + d0, DL, T.f32, D.CM_DEV)
                for hh in range(GQT)
            ]
            recs.append((ok, ms, ls, os_))
        for ok, ms, ls, os_ in recs:
            for hh in range_constexpr(GQT):
                mc = ok.select(ms[hh], fx.Float32(NEG))
                nm = fx.max(gmax[hh], mc)
                a_old = D.expf(gmax[hh] - nm)
                a_new = ok.select(D.expf(mc - nm), fx.Float32(0.0))
                tot[hh] = tot[hh] * a_old + ls[hh] * a_new
                for e in range_constexpr(DL):
                    acc[hh][e] = acc[hh][e] * a_old + os_[hh][e] * a_new
                gmax[hh] = nm
    ro = D.rsrc(o)
    for hh in range_constexpr(GQT):
        h = j * (NQ // geo.NKV) + g * GQT + hh
        inv = fx.Float32(1.0) / tot[hh]
        for e0 in range_constexpr(0, DL, 4):
            D.st(
                fx.Vector.from_elements(
                    [acc[hh][e0 + e] * inv for e in range(4)], fx.Float32
                ),
                ro,
                m * NQ * HD + h * HD + d0 + e0,
                D.CM_DEV,
            )
