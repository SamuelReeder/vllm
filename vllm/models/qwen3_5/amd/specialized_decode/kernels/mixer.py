# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token mixers of the megakernel: the Gated DeltaNet core and full attention.

Both run as wave tasks between two grid barriers (inputs: the fp32 output of the
in_proj / qkv GEMV; outputs: an fp32 vector the next GEMV prologue quantizes).

GDN task (row m, value head h, chunk c of 16 value rows), one wave. The in_proj
epilogue (stages.EpiConv) has already applied the causal conv + SiLU to the q/k/v
channels and shifted the conv state.
  lanes 0..31: l2norm of q_j (j = h // 3), 4 channels a lane; lanes 32..63: k_j.
  q, k -> wave LDS. lanes 0..15: value channel 16 c + lane; lanes 16..31: z.
  recurrence: load i of lane l = the state's [16][128] floats 4 (64 i + l) .. + 3 (one
  coalesced KiB per load; row 2 i + l // 32); row sums over k by a reduce-scatter
  butterfly (rs_sum8), after which lane l holds row r(l) (4 lanes a row).
  y[m][128 h + 16 c + r] = o * norm_w * silu(z) (not yet divided by the head RMS);
  ssq[m][h][c] = sum of o^2 over the 16 rows (the out_proj prologue finishes the norm).

Attention task (row m, kv head j, chunk ci of CH positions inside one KV block), one
wave:
  q (6 heads) and k_new: zero-centred RMSNorm over 256, NeoX RoPE on dims 0..63.
  scores (lane = position) -> partial softmax (max, sum) -> P V (lane = 4 dims).
  The chunk holding position pos writes k_new, v_new to the cache (bf16).
  Partial record [m][j][ci]: max[6], sum[6], o[6][256] fp32. The last arriver of
  (m, j) merges the chunks and writes o * sigmoid(gate) -> og[m][6144].
"""

from __future__ import annotations

import math

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import dev as D
from .dims import DK, EPS, HD, NKV, NQ

CH = 64  # attention positions per task
PVB = 16  # P V: positions per load batch
REC = 2 * 8 + 6 * HD  # partial record floats (max[6] sum[6] padded to 16, o[6][256])
GQ = NQ // NKV  # 6 q heads per kv head


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
    d = D.TRACE["d"]
    NVH, INP_N = d.nvh, d.inp_n
    Q_OFF, K_OFF, V_OFF, Z_OFF, B_OFF, A_OFF = (
        d.q_off,
        d.k_off,
        d.v_off,
        d.z_off,
        d.b_off,
        d.a_off,
    )
    D.PROBE["on"] = D.PROBE.get("stage") == "gdn"
    D.probe(c, 4)
    m = task // (NVH * 8)
    h = task % (NVH * 8) // 8
    ch = task % 8
    j = h // 3
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
    ss = D.xor_sum(ss, lane, (1, 2, 4, 8, 16, 32))
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


def bf16r(v):
    """Round f32 to bf16 and back (the KV cache precision)."""
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def pack_bf2(a, b):
    return (
        fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


@D.traced
def attn_task(
    c, task, cfg, y_qkv, qn_w, kn_w, rope_cs, kv, bt, slotp, pos_ptr, rec, cnt, og, wl
):
    """One (m, j, ci) task; wl: this wave's LDS (GQ * HD + 2 * HD // 2 floats).
    Chunk ci covers positions [p0, pe) inside KV block b = ci // cpb (see kernel.Cfg).
    Returns nothing; the last arriver of (m, j) writes og."""
    lane = c["lane"]
    NKV = D.TRACE["d"].nkv
    D.PROBE["on"] = D.PROBE.get("stage") == "attn"
    ncmax, cpb, B = cfg.ncmax, cfg.cpb, cfg.block
    m = task // (NKV * ncmax)
    j = task % (NKV * ncmax) // ncmax
    ci = task % ncmax
    pos = D.uniform(D.ld(D.rsrc(pos_ptr), m))
    slot = D.uniform(D.ld(D.rsrc(slotp), m))
    b = ci // cpb
    p0 = b * B + ci % cpb * CH
    if (p0 <= pos) & (slot >= 0):
        pe = fx.min(p0 + CH, (b + 1) * B)
        blk = D.uniform(D.ld(D.rsrc(bt), m * cfg.maxblocks + b))
        base = kv + (fx.Int64(blk) * cfg.kv_s0 + j * cfg.kv_s1) * 2
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
            cfg,
            y_qkv,
            qn_w,
            kn_w,
            rope_cs,
            base,
            rec,
            cnt,
            og,
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
    cfg,
    y_qkv,
    qn_w,
    kn_w,
    rope_cs,
    base,
    rec,
    cnt,
    og,
    wl,
):
    """base: the address of (block of p0, head j); pb: the block's first position.
    Plain layout: position p's K is at base + (p - pb) kv_s2 bf16 elements, its V kv_sv
    later. x-packed layout (vLLM's ROCM_ATTN): the head's K is [HD / 8][block][8] and
    its V, kv_sv elements later, [HD][block]."""
    d = D.TRACE["d"]
    NQ, NKV, QKV_N = d.nq, d.nkv, d.qkv_n
    ry = D.rsrc(y_qkv)
    ybase = m * QKV_N
    d0 = 4 * lane
    rk = D.rsrc(base)
    s2, sv, B, xp = cfg.kv_s2, cfg.kv_sv, cfg.block, cfg.kv_xpack
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
    # q of the 6 heads -> LDS q[h][256]
    for hh in range_constexpr(GQ):
        h = j * GQ + hh
        qv = D.ld(ry, ybase + h * 2 * HD + d0, 4, T.f32, D.CM_DEV)
        q = rms_rope(lane, [qv[e] for e in range(4)], qw, cs, True)
        fx.ptr_store(fx.Vector.from_elements(q, fx.Float32), wl + (hh * HD + d0))
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
    _store_i32x2(wl, GQ * HD + d0 // 2, kw2)
    _store_i32x2(wl, GQ * HD + HD // 2 + d0 // 2, vw2)
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
        qv = _load_f4(
            wl, fx.min(nq, GQ - 1) * HD + 32 * (i // 2) + 8 * gq + 4 * (i % 2)
        )
        w2 = fx.Vector.from_elements(
            [pack_bf2(qv[0], qv[1]), pack_bf2(qv[2], qv[3])], fx.Int32
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
        mh = D.xor_max(s_, lane, (1, 2, 4, 8, 16, 32))
        ph = live.select(D.expf(s_ - mh), fx.Float32(0.0))
        mx.append(mh)
        sm.append(D.xor_sum(ph, lane, (1, 2, 4, 8, 16, 32)))
        pr.append(ph)
    # p[h][64] -> LDS (over the q region: q is no longer needed)
    for hh in range_constexpr(GQ):
        fx.ptr_store(pr[hh], wl + (hh * CH + lane))
    # ---- P V: lane -> dims d0..d0+3
    # positions past pos have p = 0 (their address is clamped to pos); loads go out in
    # batches of PVB so their latencies overlap
    o = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(GQ)]
    vl_ = _load_i32x2(wl, GQ * HD + HD // 2 + d0 // 2)
    # (the x-packed layout loads only batches inside the chunk: its last chunk of a
    # block may be shorter than CH)
    nbat = CH // PVB
    if const_expr(xp):
        nbat = (lastp - p0) // PVB + 1
    for b in range(0, nbat):
        o = _pv_batch(wl, b, p0, pos, vl_, vload(b), o)
    D.probe(c, 7)
    # ---- partial record
    rr = D.rsrc(rec)
    ncmax = cfg.ncmax
    rbase = ((m * NKV + j) * ncmax + ci) * REC
    if lane < GQ:
        D.st(_sel6(lane, mx), rr, rbase + lane, D.CM_DEV)
        D.st(_sel6(lane, sm), rr, rbase + 8 + lane, D.CM_DEV)
    for hh in range_constexpr(GQ):
        D.st(o[hh], rr, rbase + 16 + hh * HD + d0, D.CM_DEV)
    # ---- last arriver of (m, j) merges the chunks
    rocm_wait()
    nchunk = (
        pos // cfg.block * cfg.cpb + pos % cfg.block // CH + 1
    )  # chunks up to pos's
    old = fx.Int32(0)
    if lane == 0:
        old = D.atomic_add_dev(cnt + (m * NKV + j) * 4, 1)
    old = D.uniform(old)
    D.probe(c, 8)
    if (old + 1) % nchunk == 0:
        _attn_merge(lane, d, m, j, nchunk, ncmax, ry, ybase, rr, og)
        D.probe(c, 9)


@D.traced
def _score_tile(wl, t, p0, pos, gq, nq, qa, kt):
    """Scores of positions p0 + 16 t .. + 15 for the GQ heads -> LDS s[h][16 t + n]. The
    new position's k comes from LDS (bf16 pairs at GQ HD floats), the rest from the
    cache."""
    isnew = (p0 + 16 * t + nq) == pos
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for u in range_constexpr(8):
        kl = _load_i32x4(wl, GQ * HD + (32 * u + 8 * gq) // 2)
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


def rocm_wait():
    rocdl.s_waitcnt(vmcnt=0)


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


MERGE_B = 2  # chunks per merge batch


@D.traced
def _attn_merge(lane, d, m, j, nchunk, ncmax, ry, ybase, rr, og):
    NQ, NKV = d.nq, d.nkv
    d0 = 4 * lane
    base = (m * NKV + j) * ncmax * REC
    gmax = [fx.Float32(-3.0e38)] * GQ
    tot = [fx.Float32(0.0)] * GQ
    acc = [[fx.Float32(0.0)] * 4 for _ in range(GQ)]
    for b0 in range(0, (nchunk + MERGE_B - 1) // MERGE_B):
        recs = []
        for k in range_constexpr(MERGE_B):
            cidx = fx.min(b0 * MERGE_B + k, nchunk - 1)
            ok = (b0 * MERGE_B + k) < nchunk
            rb = base + cidx * REC
            ms = D.ld(rr, rb, 4, T.f32, D.CM_DEV), D.ld(rr, rb + 4, 4, T.f32, D.CM_DEV)
            ls = (
                D.ld(rr, rb + 8, 4, T.f32, D.CM_DEV),
                D.ld(rr, rb + 12, 4, T.f32, D.CM_DEV),
            )
            os_ = [
                D.ld(rr, rb + 16 + hh * HD + d0, 4, T.f32, D.CM_DEV) for hh in range(GQ)
            ]
            recs.append((ok, ms, ls, os_))
        for ok, ms, ls, os_ in recs:
            for hh in range_constexpr(GQ):
                mc = ms[hh // 4][hh % 4]
                lc = ls[hh // 4][hh % 4]
                mc = ok.select(mc, fx.Float32(-3.0e38))
                nm = fx.max(gmax[hh], mc)
                a_old = D.expf(gmax[hh] - nm)
                a_new = ok.select(D.expf(mc - nm), fx.Float32(0.0))
                tot[hh] = tot[hh] * a_old + lc * a_new
                for e in range_constexpr(4):
                    acc[hh][e] = acc[hh][e] * a_old + os_[hh][e] * a_new
                gmax[hh] = nm
    ro = D.rsrc(og)
    for hh in range_constexpr(GQ):
        h = j * GQ + hh
        gt = D.ld(ry, ybase + h * 2 * HD + HD + d0, 4, T.f32, D.CM_DEV)
        inv = fx.Float32(1.0) / tot[hh]
        vals = [acc[hh][e] * inv * D.sigmoid(gt[e]) for e in range(4)]
        D.st(
            fx.Vector.from_elements(vals, fx.Float32),
            ro,
            m * NQ * HD + h * HD + d0,
            D.CM_DEV,
        )
