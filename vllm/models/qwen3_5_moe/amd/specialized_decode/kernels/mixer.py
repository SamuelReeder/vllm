# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token mixers of the megakernel: the Gated DeltaNet core and full attention.

Both run as wave tasks between two grid barriers (inputs: the fp32 output of the
in_proj / qkv GEMV; outputs: an fp32 vector the next GEMV prologue quantizes).

GDN task (row m, value head h, chunk c of 16 value rows), one wave. The in_proj
epilogue (stages.EpiConv) has already applied the causal conv + SiLU to the q/k/v
channels and shifted the conv state.
  lanes 0..31: l2norm of q_j (j = h // 2), 4 channels a lane; lanes 32..63: k_j.
  q, k -> wave LDS. lanes 0..15: value channel 16 c + lane; lanes 16..31: z.
  recurrence: load i of lane l = the state's [16][128] floats 4 (64 i + l) .. + 3 (one
  coalesced KiB per load; row 2 i + l // 32); row sums over k by a reduce-scatter
  butterfly (rs_sum8), after which lane l holds row r(l) (4 lanes a row).
  y[m][128 h + 16 c + r] = o * norm_w * silu(z) (not yet divided by the head RMS);
  ssq[m][h][c] = sum of o^2 over the 16 rows (the out_proj prologue finishes the norm).

Attention task (row m, kv head j, chunk ci of CH positions inside one KV block), one
wave:
  q (8 heads) and k_new: zero-centred RMSNorm over 256, NeoX RoPE on dims 0..63.
  scores (lane = position) -> partial softmax (max, sum) -> P V (lane = 4 dims).
  The chunk holding position pos writes k_new, v_new to the cache (bf16).
  Partial record [m][j][ci]: max[8], sum[8], o[8][256] fp32. The o_proj prologue
  (prologue_attn) merges the records of its head in chunk order and multiplies by
  sigmoid(gate).
"""

from __future__ import annotations

import math

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import stages as S
from .dims import (
    A_OFF,
    B_OFF,
    DK,
    EPS,
    HD,
    INP_N,
    K_OFF,
    NKH,
    NKV,
    NQ,
    NVH,
    Q_OFF,
    QKV_N,
    V_OFF,
    Z_OFF,
)
from .mkcore import dev as D
from .mkcore import stagelib as SL
from .mkcore.dev import bf16r, pack_bf2

CH = 64  # attention positions per task
PVB = 16  # P V: positions per load batch
GQ = NQ // NKV  # 8 q heads per kv head
REC = 16 + GQ * HD  # partial record floats (max[8] sum[8], o[8][256])
QW = GQ * HD // 2  # wave LDS floats of q (bf16 pairs); then k_new, v_new (bf16 pairs)
# kernel.Cfg.rt_geom 2: the Cfg fields not compiled in, and the run-time words the
# attention stages load instead (i32 words of glob from kernel.G_GEOM, in this order)
RT_FIELDS = ("maxlen", "maxblocks")
GEO = ("ncmax", "maxblocks")
GEO_ATT = ("block", "cpb", "ncmax", "maxblocks", "kv_s0", "kv_s1", "kv_s2", "kv_sv")


class Geo:
    """The attention stages' cache geometry: cfg's constants, and with cfg.rt_geom the
    GEO words of glob from i32 word `word0`, loaded where attn is called (call it in the
    scope that uses the values). cfg: the compiled Cfg."""

    def __init__(self, cfg, rg, word0):
        self.cfg, self.rg, self.word0 = cfg, rg, word0

    def attn(self):
        """attn_task's and prologue_attn's geometry: dict of GEO_ATT plus kv_xpack."""
        g = {n: getattr(self.cfg, n) for n in GEO_ATT}
        if self.cfg.rt_geom:
            for k, n in enumerate(GEO):
                g[n] = D.uniform(D.ld(self.rg, self.word0 + k))
        return dict(g, kv_xpack=self.cfg.kv_xpack)


# ------------------------------------------------------------------ GDN
@D.traced
def gdn_task(
    c, M, task, y_in, a_log, dt_bias, norm_w, ssm, sidx, ssm_s0, y_out, ssq_out, wl
):
    """One (m, h, chunk) task. y_in: the in_proj output with the causal conv + SiLU
    already applied to the q/k/v channels (S.EpiConv). ssm: the layer's SSM state of
    slot 0, slot s at ssm + s ssm_s0 bytes ([NVH][DV][DK] fp32); row m's slot is
    sidx[m] (0: a padding row, whose state is not written). wl: this wave's LDS (256
    floats: q[128] k[128])."""
    lane = c["lane"]
    D.PROBE["on"] = D.PROBE.get("stage") == "gdn"
    D.probe(c, 4)
    m = task // (NVH * 8)
    h = task % (NVH * 8) // 8
    ch = task % 8
    j = h // (NVH // NKH)
    ry = D.rsrc(y_in)
    ybase = m * INP_N
    # ---- q_j / k_j: lane l < 32 -> q channels 4l..4l+3, l >= 32 -> k channels
    # 4(l-32)..
    isk = lane >= 32
    ci = (lane % 32) * 4
    chan = isk.select(K_OFF + j * DK + ci, Q_OFF + j * DK + ci)
    yq = D.ld(ry, ybase + chan, 4, T.f32, D.CM_DEV)
    yv = [yq[e] for e in range(4)]
    ss = yv[0] * yv[0] + yv[1] * yv[1] + yv[2] * yv[2] + yv[3] * yv[3]
    ss = D.xor_sum(ss, lane, (1, 2, 4, 8, 16))
    rn = D.rsq(ss + fx.Float32(1e-6))
    qs = isk.select(fx.Float32(1.0), fx.Float32(1.0 / math.sqrt(DK)))
    for e in range_constexpr(4):
        fx.ptr_store(yv[e] * rn * qs, wl + (isk.select(DK, 0) + ci + e))
    # ---- v / z of this chunk: lane < 16 -> v channel, 16..31 -> z
    vr = lane % 16
    vch = V_OFF + h * 128 + ch * 16 + vr
    vval = D.ldf(ry, ybase + vch, cm=D.CM_DEV)
    zval = D.ldf(ry, ybase + Z_OFF + h * 128 + ch * 16 + vr, cm=D.CM_DEV)
    bval = D.ldf(ry, ybase + B_OFF + h, cm=D.CM_DEV)
    aval = D.ldf(ry, ybase + A_OFF + h, cm=D.CM_DEV)
    alog = D.bf_lo(D.ld(D.rsrc(a_log), h // 2) >> ((h % 2) * 16))
    dtb = D.bf_lo(D.ld(D.rsrc(dt_bias), h // 2) >> ((h % 2) * 16))
    xa = aval + dtb
    sp = (xa > fx.Float32(20.0)).select(xa, D.logf1p_exp(xa))
    decay = D.expf(-D.expf(alog) * sp)
    beta = D.sigmoid(bval)
    # q/k reach the other lanes through this wave's LDS: a wave's LDS ops run in order
    D.probe(c, 5)
    # ---- recurrence. Load i of lane l is the 16 B at float 4 (64 i + l) of the chunk's
    # [16][128] state (one contiguous KiB per instruction): S[2 i + l // 32][kc .. kc +
    # 3] with kc = 4 (l % 32). Row sums over k then span 32 lanes: a reduce-scatter
    # butterfly (lane bits 4, 3, 2) leaves lane l the sum of instruction i(l) = 4 b4 + 2
    # b3 + b2, and lane bits 1, 0 finish it (see rs_sum8).
    slot = D.uniform(D.ld(D.rsrc(sidx), m))
    rs = D.rsrc(ssm + fx.Int64(slot) * ssm_s0)
    sbase = (h * 128 + ch * 16) * DK + 4 * lane
    S = [D.ld(rs, sbase + 256 * i, 4, T.f32, D.CM_NT) for i in range(8)]
    kc = 4 * (lane % 32)
    kk = [fx.ptr_load(wl + (DK + kc + e)) for e in range(4)]
    qq = [fx.ptr_load(wl + (kc + e)) for e in range(4)]
    D.probe(c, 6)
    r = (
        2 * (((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1))
        + lane // 32
    )  # lane's row
    kv = rs_sum8(
        [
            S[i][0] * kk[0] + S[i][1] * kk[1] + S[i][2] * kk[2] + S[i][3] * kk[3]
            for i in range(8)
        ],
        lane,
    )
    vrow = D.gather_f(vval, r)  # v of row r lives in lane r
    delta = (vrow - kv * decay) * beta
    Sn, po = [], []
    for i in range_constexpr(8):
        # row 2 i + l // 32's delta lives in the lanes with i(l') = i (and l' // 32 = l
        # // 32)
        di = D.gather_f(
            delta,
            (lane & 32) | ((i >> 2) & 1) << 4 | ((i >> 1) & 1) << 3 | (i & 1) << 2,
        )
        nv = [S[i][e] * decay + kk[e] * di for e in range(4)]
        Sn.append(fx.Vector.from_elements(nv, fx.Float32))
        po.append(nv[0] * qq[0] + nv[1] * qq[1] + nv[2] * qq[2] + nv[3] * qq[3])
    if slot != 0:
        for i in range_constexpr(8):
            D.st(Sn[i], rs, sbase + 256 * i, D.CM_NT)
    o = rs_sum8(po, lane)
    zrow = D.gather_f(zval, 16 + r)
    nwr = D.bf_lo(
        D.ld(D.rsrc(norm_w), (ch * 16 + r) // 2) >> (((ch * 16 + r) % 2) * 16)
    )
    y = o * nwr * D.silu(zrow)
    first = (lane & 3) == 0  # one of the 4 lanes of each row
    s2 = D.xor_sum(first.select(o * o, fx.Float32(0.0)), lane, (4, 8, 16, 32))
    if first:
        D.st(y, D.rsrc(y_out), m * NVH * 128 + h * 128 + ch * 16 + r, D.CM_DEV)
    if lane == 0:
        D.st(s2, D.rsrc(ssq_out), (m * NVH + h) * 8 + ch, D.CM_DEV)
    D.probe(c, 7)


def rs_sum8(vals, lane):
    """vals[i] (i < 8) summed over the 32 lanes of the lane's half-wave, lane l getting
    the sum of i = 4 b4 + 2 b3 + b2 (bits of l). Reduce-scatter steps 16, 8, 4 halve the
    values a lane holds (send the other half to the partner, add the partner's copy of
    the kept half), then steps 2, 1 sum the remaining value."""
    v = list(vals)
    for msk in (16, 8, 4):
        hi = (lane & msk) != 0
        n = len(v) // 2
        nv = []
        for k in range(n):
            keep = hi.select(v[k + n], v[k])
            send = hi.select(v[k], v[k + n])
            nv.append(keep + D.gather_f(send, lane ^ msk))
        v = nv
    return D.xor_sum(v[0], lane, (1, 2))


# ------------------------------------------------------------------ attention
def rms_rope(lane, vals, w_word, cs, rope):
    """Zero-centred RMSNorm over 256 dims spread 4 a lane (dims 4 lane + e), then NeoX
    RoPE on dims 0..63 (pairs d, d + 32 live in lanes l, l ^ 8). w_word: the lane's 4
    bf16 norm weights (2 dwords); cs: (cos, sin) of the lane's 4 dims (dims < 32
    index)."""
    ss = vals[0] * vals[0] + vals[1] * vals[1] + vals[2] * vals[2] + vals[3] * vals[3]
    ss = D.wave_sum(ss)
    rn = D.rsq(ss / fx.Float32(float(HD)) + fx.Float32(EPS))
    wf = [
        D.bf_lo(w_word[0]),
        D.bf_hi(w_word[0]),
        D.bf_lo(w_word[1]),
        D.bf_hi(w_word[1]),
    ]
    x = [vals[e] * rn * (fx.Float32(1.0) + wf[e]) for e in range(4)]
    if not rope:
        return x
    cos, sin = cs
    lo = lane < 8  # dims 0..31: x1; lanes 8..15: x2
    inr = lane < 16
    out = []
    for e in range_constexpr(4):
        partner = D.gather_f(x[e], lane ^ 8)
        r1 = x[e] * cos[e] - partner * sin[e]  # d < 32: x1 cos - x2 sin
        r2 = x[e] * cos[e] + partner * sin[e]  # d >= 32: x2 cos + x1 sin
        out.append(inr.select(lo.select(r1, r2), x[e]))
    return out


@D.traced
def attn_task(c, task, g, y_qkv, qn_w, kn_w, rope_cs, kv, bt, slotp, pos_ptr, rec, wl):
    """One (m, j, ci) task; wl: this wave's LDS (QW + HD floats); g: the cache geometry
    (Geo.attn). Chunk ci covers positions [p0, pe) inside KV block b = ci // cpb (see
    kernel.Cfg). Writes the chunk's partial record (merged by prologue_attn)."""
    lane = c["lane"]
    D.PROBE["on"] = D.PROBE.get("stage") == "attn"
    ncmax, cpb, B = g["ncmax"], g["cpb"], g["block"]
    m = task // (NKV * ncmax)
    j = task % (NKV * ncmax) // ncmax
    ci = task % ncmax
    pos = D.uniform(D.ld(D.rsrc(pos_ptr), m))
    slot = D.uniform(D.ld(D.rsrc(slotp), m))
    b = ci // cpb
    p0 = b * B + ci % cpb * CH
    if (p0 <= pos) & (slot >= 0):
        pe = fx.min(p0 + CH, (b + 1) * B)
        blk = D.uniform(D.ld(D.rsrc(bt), m * g["maxblocks"] + b))
        base = kv + (fx.Int64(blk) * g["kv_s0"] + j * g["kv_s1"]) * 2
        _attn_chunk(
            c,
            lane,
            m,
            j,
            ci,
            pos,
            p0,
            pe,
            b * B,
            g,
            y_qkv,
            qn_w,
            kn_w,
            rope_cs,
            base,
            rec,
            wl,
        )


@D.traced
def _attn_chunk(
    c,
    lane,
    m,
    j,
    ci,
    pos,
    p0,
    pe,
    pb,
    g,
    y_qkv,
    qn_w,
    kn_w,
    rope_cs,
    base,
    rec,
    wl,
):
    """base: the address of (block of p0, head j); pb: the block's first position.
    Plain layout: position p's K is at base + (p - pb) kv_s2 bf16 elements, its V kv_sv
    later. x-packed layout (vLLM's ROCM_ATTN): the head's K is [HD / 8][block][8] and
    its V, kv_sv elements later, [HD][block]."""
    ry = D.rsrc(y_qkv)
    ybase = m * QKV_N
    d0 = 4 * lane
    rk = D.rsrc(base)
    s2, sv, B, xp = g["kv_s2"], g["kv_sv"], g["block"], g["kv_xpack"]
    lastp = fx.min(pos, pe - 1)  # the chunk's last live position
    p = p0 + lane
    live = p <= lastp
    gq, nq = lane // 16, lane % 16  # MFMA lane coordinates

    def ktile(t):
        """B operand rows of score tile t: position p0 + 16 t + nq (clamped to lastp).
        Load u of lane group gq: dims 32 u + 8 gq .. + 7 (the 4 groups read 64
        contiguous bytes of a row); k-step 2 u + h = dims 32 u + 8 gq + 4 h .. + 3."""
        pt = fx.min(p0 + 16 * t + nq, lastp)
        if const_expr(xp):
            return [D.ld(rk, (4 * u + gq) * 4 * B + (pt - pb) * 4, 4) for u in range(8)]
        return [D.ld(rk, ((pt - pb) * s2 + 32 * u + 8 * gq) // 2, 4) for u in range(8)]

    def vload(b):
        """[k] -> (dims d0, d0 + 1 | d0 + 2, d0 + 3) bf16 pairs of position p0 + PVB b +
        k; positions past lastp: the plain layout loads lastp's, the x-packed layout
        zeros (its rows of PVB positions are loaded whole, and stale slots may hold
        NaNs)."""
        if const_expr(not xp):
            return [
                D.ld(
                    rk, ((fx.min(p0 + b * PVB + k, lastp) - pb) * s2 + sv + d0) // 2, 2
                )
                for k in range(PVB)
            ]
        q0 = p0 + b * PVB - pb
        rows = [
            [
                D.ld(rk, (sv + (d0 + e) * B + q0) // 2 + 4 * h, 4)
                for h in range(PVB // 8)
            ]
            for e in range(4)
        ]
        out = []
        for k in range_constexpr(PVB):
            hw = [
                (rows[e][k // 8][k % 8 // 2] >> (16 * (k % 2))) & 0xFFFF
                for e in range(4)
            ]
            ok = (p0 + b * PVB + k) <= lastp
            z = fx.Int32(0)
            out.append(
                [
                    ok.select(hw[0] | (hw[1] << 16), z),
                    ok.select(hw[2] | (hw[3] << 16), z),
                ]
            )
        return out

    kt = ktile(0)  # independent of this step's q/k/v
    D.probe(c, 4)
    # rope table [maxlen][64] fp32: cos[0..31], sin[0..31] of position pos
    rcs = D.rsrc(rope_cs)
    di = d0 % 32
    cosv = D.ld(rcs, pos * 64 + di, 4, T.f32)
    sinv = D.ld(rcs, pos * 64 + 32 + di, 4, T.f32)
    cs = ([cosv[e] for e in range(4)], [sinv[e] for e in range(4)])
    qw = D.ld(D.rsrc(qn_w), d0 // 2, 2)
    kw = D.ld(D.rsrc(kn_w), d0 // 2, 2)
    # q of the 8 heads -> LDS q[h][256] as bf16 pairs (the precision of the score MFMA)
    for hh in range_constexpr(GQ):
        h = j * GQ + hh
        qv = D.ld(ry, ybase + h * 2 * HD + d0, 4, T.f32, D.CM_DEV)
        q = rms_rope(lane, [qv[e] for e in range(4)], qw, cs, True)
        q2 = fx.Vector.from_elements(
            [pack_bf2(q[0], q[1]), pack_bf2(q[2], q[3])], fx.Int32
        )
        _store_i32x2(wl, (hh * HD + d0) // 2, q2)
    # k_new / v_new (bf16-rounded, as the cache holds them) -> LDS kn[256] vn[256] (bf16
    # pairs)
    kv_ = D.ld(ry, ybase + 2 * NQ * HD + j * HD + d0, 4, T.f32, D.CM_DEV)
    vv_ = D.ld(ry, ybase + 2 * NQ * HD + NKV * HD + j * HD + d0, 4, T.f32, D.CM_DEV)
    kn = rms_rope(lane, [kv_[e] for e in range(4)], kw, cs, True)
    kw2 = fx.Vector.from_elements(
        [pack_bf2(kn[0], kn[1]), pack_bf2(kn[2], kn[3])], fx.Int32
    )
    vw2 = fx.Vector.from_elements(
        [pack_bf2(vv_[0], vv_[1]), pack_bf2(vv_[2], vv_[3])], fx.Int32
    )
    _store_i32x2(wl, QW + d0 // 2, kw2)
    _store_i32x2(wl, QW + HD // 2 + d0 // 2, vw2)
    if (pos >= p0) & (pos < pe):
        if const_expr(xp):
            D.st(kw2, rk, ((d0 // 8) * 8 * B + (pos - pb) * 8 + d0 % 8) // 2, D.CM_DEV)
            for e in range_constexpr(4):
                D.st(
                    bf16r(vv_[e]).to(fx.BFloat16),
                    rk,
                    sv + (d0 + e) * B + pos - pb,
                    D.CM_DEV,
                )
        else:
            D.st(kw2, rk, ((pos - pb) * s2 + d0) // 2, D.CM_DEV)
            D.st(vw2, rk, ((pos - pb) * s2 + sv + d0) // 2, D.CM_DEV)
    D.probe(c, 5)
    # ---- scores on the MFMA (bf16 16x16x16): S^T tile [16 head rows][16 positions].
    # The k order is permuted alike in A and B (see ktile): the sum over k does not
    # depend on it. Rows >= GQ are zero.
    qa = []
    for i in range_constexpr(16):
        w2 = _load_i32x2(
            wl, (fx.min(nq, GQ - 1) * HD + 32 * (i // 2) + 8 * gq + 4 * (i % 2)) // 2
        )
        z2 = fx.Vector.from_elements([fx.Int32(0), fx.Int32(0)], fx.Int32)
        qa.append((nq < GQ).select(w2, z2))
    # (q is in registers now: its LDS region takes the scores / probabilities)
    for t in range(0, D.opaque_s(CH // 16)):  # opaque trip count: LLVM would unroll
        kn_ = ktile(fx.min(t + 1, CH // 16 - 1))
        _score_tile(wl, t, p0, pos, gq, nq, qa, kt)
        kt = kn_
    D.probe(c, 6)
    scale = 1.0 / math.sqrt(HD)
    mx, sm, pr = [], [], []
    for hh in range_constexpr(GQ):
        s_ = live.select(
            fx.ptr_load(wl + (hh * CH + lane)) * scale, fx.Float32(-3.0e38)
        )
        mh = D.wave_max(s_)
        ph = live.select(D.expf(s_ - mh), fx.Float32(0.0))
        mx.append(mh)
        sm.append(D.wave_sum(ph))
        pr.append(ph)
    # p[h][64] -> LDS (over the q region: q is no longer needed)
    for hh in range_constexpr(GQ):
        fx.ptr_store(pr[hh], wl + (hh * CH + lane))
    # ---- P V: lane -> dims d0..d0+3
    # positions past pos have p = 0 (their address is clamped to pos); loads go out in
    # batches of PVB so their latencies overlap
    o = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(GQ)]
    vl_ = _load_i32x2(wl, QW + HD // 2 + d0 // 2)
    # (the x-packed layout loads only batches inside the chunk: its last chunk of a
    # block may be shorter than CH)
    nbat = CH // PVB
    if const_expr(xp):
        nbat = (lastp - p0) // PVB + 1
    for b in range(0, nbat):
        o = _pv_batch(wl, b, p0, pos, vl_, vload(b), o)
    D.probe(c, 7)
    # ---- partial record (merged by the o_proj prologue, prologue_attn)
    rr = D.rsrc(rec)
    rbase = ((m * NKV + j) * g["ncmax"] + ci) * REC
    if lane < GQ:
        D.st(_sel6(lane, mx), rr, rbase + lane, D.CM_DEV)
        D.st(_sel6(lane, sm), rr, rbase + 8 + lane, D.CM_DEV)
    for hh in range_constexpr(GQ):
        D.st(o[hh], rr, rbase + 16 + hh * HD + d0, D.CM_DEV)
    D.probe(c, 8)


@D.traced
def _score_tile(wl, t, p0, pos, gq, nq, qa, kt):
    """Scores of positions p0 + 16 t .. + 15 for the GQ heads -> LDS s[h][16 t + n]. The
    new position's k comes from LDS (bf16 pairs at QW floats), the rest from the
    cache."""
    isnew = (p0 + 16 * t + nq) == pos
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for u in range_constexpr(8):
        kl = _load_i32x4(wl, QW + (32 * u + 8 * gq) // 2)
        kw = [isnew.select(kl[e], kt[u][e]) for e in range(4)]
        for h in range_constexpr(2):
            a = qa[2 * u + h].bitcast(fx.Int16)
            b = fx.Vector.from_elements([kw[2 * h], kw[2 * h + 1]], fx.Int32).bitcast(
                fx.Int16
            )
            acc = fx.Vector(
                rocdl.mfma_f32_16x16x16bf16_1k(T.vec(4, T.f32), [a, b, acc, 0, 0, 0])
            )
    # D[row 4 gq + r][col nq]: head 4 gq + r, position 16 t + nq
    for r in range_constexpr(4):
        if 4 * gq + r < GQ:
            fx.ptr_store(acc[r], wl + ((4 * gq + r) * CH + 16 * t + nq))


@D.traced
def _pv_batch(wl, b, p0, pos, vl_, vrs, o):
    """O += p V over positions p0 + PVB b + k (p = 0 past pos)."""
    o = [[o[hh][e] for e in range(4)] for hh in range(GQ)]
    for k in range_constexpr(PVB):
        pp = b * PVB + k
        nw = (p0 + pp) == pos
        w0 = nw.select(vl_[0], vrs[k][0])
        w1 = nw.select(vl_[1], vrs[k][1])
        vf = [D.bf_lo(w0), D.bf_hi(w0), D.bf_lo(w1), D.bf_hi(w1)]
        for hh in range_constexpr(GQ):
            ph = fx.ptr_load(wl + (hh * CH + pp))
            for e in range_constexpr(4):
                o[hh][e] = D.fma(ph, vf[e], o[hh][e])
    return [fx.Vector.from_elements(o[hh], fx.Float32) for hh in range(GQ)]


def _sel6(lane, vals):
    v = vals[0]
    for i in range(1, len(vals)):
        v = (lane == i).select(vals[i], v)
    return v


def _store_i32x2(base_f32, idx, v):
    """Store 2 i32 at float-index idx of an f32 LDS pointer."""
    fx.ptr_store(v.bitcast(fx.Float32), base_f32 + idx)


def _load_i32x2(base_f32, idx):
    return fx.Vector(
        fx.ptr_load(base_f32 + idx, result_type=fx.Vector.make_type(2, fx.Float32))
    ).bitcast(fx.Int32)


def _load_f4(base_f32, idx):
    return fx.Vector(
        fx.ptr_load(base_f32 + idx, result_type=fx.Vector.make_type(4, fx.Float32))
    )


def _load_i32x4(base_f32, idx):
    return fx.Vector(
        fx.ptr_load(base_f32 + idx, result_type=fx.Vector.make_type(4, fx.Float32))
    ).bitcast(fx.Int32)


MERGE_RB = 4  # chunk records per load batch of prologue_attn


@D.traced
def prologue_attn(c, g, xq, xs, geo, rec, y_qkv, pos_ptr, slotp):
    """o_proj prologue: LDS x <- the attention output of the CTA's chunk (one head:
    g.Kc == HD), quantized. Thread item (row t, dims d .. d + 7) merges the head's
    chunk records in chunk order (online softmax: running max, rescaled sum and o),
    then x = o / sum * sigmoid(gate). Rows without a KV slot (padding) give x = 0.
    geo: the cache geometry (Geo.attn)."""
    assert g.Kc == HD and g.slots == 1
    M, tid, NT = g.M, c["tid"], S.NT
    SL.zero_xs_tail(c, g, xs)
    rr, ry, rp, rs = D.rsrc(rec), D.rsrc(y_qkv), D.rsrc(pos_ptr), D.rsrc(slotp)
    h = fx.min(g.chunk0(c["bid"]), g.C - 1)  # the CTA's head
    j, hh = h // GQ, h % GQ

    def nchunk(t):
        pos = D.ld(rp, t)
        B = geo["block"]
        n = pos // B * geo["cpb"] + pos % B // CH + 1  # chunks up to pos's
        return (D.ld(rs, t) >= 0).select(n, fx.Int32(0))

    nmax = fx.Int32(0)
    for t in range_constexpr(M):
        nmax = fx.max(nmax, D.uniform(nchunk(t)))
    nch = M * HD // 8
    items = []
    for i in range_constexpr((nch + NT - 1) // NT):
        q = fx.min(tid + NT * i, nch - 1)
        t = q // (HD // 8)
        d = q % (HD // 8) * 8
        gb = t * QKV_N + h * 2 * HD + HD + d
        gt = (D.ld(ry, gb, 4, T.f32, D.CM_DEV), D.ld(ry, gb + 4, 4, T.f32, D.CM_DEV))
        items.append((i, t, d, nchunk(t), ((t * NKV + j) * geo["ncmax"]) * REC, gt))
    # loop-carried state per item: [max, sum, o0 .. o7] (updated in place, as the
    # FlyDSL loop rewriter carries subscript stores)
    st = [
        fx.Float32(-3.0e38) if e == 0 else fx.Float32(0.0)
        for _ in items
        for e in range(10)
    ]
    for b0 in range(0, (nmax + MERGE_RB - 1) // MERGE_RB):
        recs = []
        for _, _, d, n, base, _ in items:
            rl = []
            for k in range_constexpr(MERGE_RB):
                ok = (b0 * MERGE_RB + k) < n
                rb = base + fx.min(b0 * MERGE_RB + k, fx.max(n - 1, 0)) * REC
                rl.append(
                    (
                        ok,
                        D.ldf(rr, rb + hh, cm=D.CM_DEV),
                        D.ldf(rr, rb + 8 + hh, cm=D.CM_DEV),
                        D.ld(rr, rb + 16 + hh * HD + d, 4, T.f32, D.CM_DEV),
                        D.ld(rr, rb + 16 + hh * HD + d + 4, 4, T.f32, D.CM_DEV),
                    )
                )
            recs.append(rl)
        for q in range_constexpr(len(items)):
            for ok, mc, lc, o0, o1 in recs[q]:
                mc = ok.select(mc, fx.Float32(-3.0e38))
                nm = fx.max(st[10 * q], mc)
                a_old = D.expf(st[10 * q] - nm)
                a_new = ok.select(D.expf(mc - nm), fx.Float32(0.0))
                st[10 * q + 1] = st[10 * q + 1] * a_old + lc * a_new
                ov = [o0[0], o0[1], o0[2], o0[3], o1[0], o1[1], o1[2], o1[3]]
                for e in range_constexpr(8):
                    st[10 * q + 2 + e] = st[10 * q + 2 + e] * a_old + ov[e] * a_new
                st[10 * q] = nm
    for q in range_constexpr(len(items)):
        i, t, d, n, _, gt = items[q]
        inv = fx.Float32(1.0) / st[10 * q + 1]
        gv = [
            gt[0][0],
            gt[0][1],
            gt[0][2],
            gt[0][3],
            gt[1][0],
            gt[1][1],
            gt[1][2],
            gt[1][3],
        ]
        x = [
            (n > 0).select(st[10 * q + 2 + e] * inv * D.sigmoid(gv[e]), fx.Float32(0.0))
            for e in range(8)
        ]
        SL.store_quant8(c, g, xq, xs, x, t, d, tid + NT * i < nch)
    gpu.barrier()
