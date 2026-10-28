# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8-27B decode megakernel: one persistent launch per decode step.

Grid = CU count, 512 threads, all CTAs co-resident. Stages are separated by
grid barriers (dev.grid_arrive / grid_wait). Per layer:

  GDN layer:  in_proj GEMV -> | GDN core tasks -> | out_proj GEMV (+h) -> |
  attn layer: qkv GEMV     -> | attention tasks -> | o_proj GEMV (+h)   -> |
  both:       gate_up GEMV -> | down GEMV (+h) -> |
then the final norm and either the LM head GEMV (bf16) -> fp32 logits, or the
normed hidden states (bf16) for a serving engine's own LM head.

Residual stream: h0 at a layer start, h1 after the mixer (both fp32 [M][H], with
their 16-column sums of squares ssq0 / ssq1). See docs/DESIGN.md.

Caches use vLLM's layouts (Cfg holds their strides):
  KV of an attention layer: bf16, K of (block, head, position) at element strides
    (kv_s0, kv_s1, kv_s2), its V kv_sv elements later; row m's block b is bt[m][b].
    vLLM's backends store a block as [NKV][block][2 HD] or [K|V][block][NKV HD]
    (TRITON_ATTN), or x-packed (ROCM_ATTN: kv_xpack, see mixer._attn_chunk).
  Each layer's table row points at the cache indices of its vLLM KV cache group
  (T_IDX, T_SLOT): the groups of a hybrid model share the cache memory, so the same
  request has different blocks and slots in each.
  GDN layer: per state slot s, the conv state [3][CONV_DIM] bf16 (oldest input first)
    at conv + s conv_s0 bytes and the SSM state [NVH][DV][DK] fp32 at ssm + s ssm_s0
    bytes (vLLM keeps both in one page per slot); row m's slot is sidx[m] (0: padding,
    not written).
  A row with slot[m] < 0 (padding) reads and writes no KV.
The barrier epoch lives on the device (word EPOCH_WORD of the flag buffer), so a
launch has no per-step host argument and can be captured in a CUDA graph.
"""

import functools
import zlib
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from . import dev as D
from . import mixer as X
from . import stages as S
from .dims import EPS, HD, Dims, H
from .gemv import Gemv, chunks

NTAB = 18  # pointer-table slots per layer
NGLOB = 18  # global pointer-table slots
WAVES = 8
NT = 64 * WAVES

# global table slots
(
    G_EMBED,
    G_LM,
    G_FNORM,
    G_ROPE,
    G_HB,
    G_SSQ,
    G_Y,
    G_Y2,
    G_HSSQ,
    G_PART,
    G_CNT,
    G_FLAGS,
    G_REC,
    G_TOK,
    G_POS,
    G_OUT,
    G_PEERS,  # TP: int64 [tp] every rank's peer buffer (PeerBuffer.addresses)
    G_RANK,  # TP: this rank (an integer, not a pointer)
) = range(18)
# per-layer slots
(
    T_MIX_T,
    T_MIX_S,
    T_OUT_T,
    T_OUT_S,
    T_GU_T,
    T_GU_S,
    T_DN_T,
    T_DN_S,
    T_IN_LN,
    T_POST_LN,
) = range(10)
T_ALOG, T_DTB, T_CONVW, T_NORMW, T_CONV, T_SSM = range(10, 16)  # GDN layers
T_QN, T_KN, T_KV = range(10, 13)  # attention layers
# The layer's cache indices, per vLLM KV cache group: a GDN layer's state slots
# sidx[M]; an attention layer's block table bt[M][maxblocks] and slot mapping slot[M].
T_IDX, T_SLOT = 16, 17

_STREAM = fx.Stream(None)  # the launch's default: the current stream
CNT_STRIDE = 65536  # i32 counters per counter region
EPOCH_WORD_LINE = 8  # the epoch word is line cus + EPOCH_WORD_LINE of the flag buffer


@dataclass(frozen=True)
class Cfg:
    """One compiled kernel: M rows on `cus` CUs, and the cache geometry it reads."""

    M: int
    cus: int
    maxlen: int  # positions < maxlen (the RoPE table's length)
    block: int  # KV positions per block
    maxblocks: int  # columns of the block table
    kv_s0: int  # KV element strides: block, head, position
    kv_s1: int
    kv_s2: int
    kv_sv: int  # elements from a K row to its V row
    kv_xpack: bool  # vLLM ROCM_ATTN's x-packed block interior (see mixer._attn_chunk)
    conv_s0: int  # bytes between the conv states of two slots
    ssm_s0: int  # bytes between the SSM states of two slots
    lm_head: bool = True  # False: write the normed hidden states instead of logits
    tp: int = 1  # tensor-parallel size (docs/DESIGN-TP2.md); the kernel holds one rank
    ar_dtype: str = "fp32"  # all-reduce partials: "fp32" pairs or "bf16" pair words
    ar_local: bool = False  # debug timing only: add the own partial twice, no peer
    chunks: tuple = ()  # ((gemv name, K chunks), ...) over gemv.chunks(tp); Packed's

    @property
    def d(self):
        return Dims(self.tp)

    @property
    def cpb(self):
        """Attention chunks per KV block: a chunk never straddles two blocks."""
        return -(-self.block // X.CH)

    @property
    def ncmax(self):
        return self.maxblocks * self.cpb

    def specs(self):
        M, cus, d = self.M, self.cus, self.d
        ch = chunks(self.tp, self.chunks)
        g = lambda name, n, k, kind="mx": Gemv(
            name,
            n,
            k,
            M,
            cus,
            WAVES,
            ch.get(name, 1),  # noqa: E731
            kind,
        ).check()
        return dict(
            inp=g("inp", d.inp_n, H),
            out=g("out", H, d.val_dim),
            qkv=g("qkv", d.qkv_n, H),
            o=g("o", H, d.nq * HD),
            gu=g("gu", 2 * d.ffn, H),
            down=g("down", H, d.ffn),
            lm=g("lm", d.vocab, H, kind="bf16"),
        )

    def part_floats(self):
        return max(s.part_floats() for s in self.specs().values())


# Peer buffer (TP > 1): two halves for the layer all-reduces (out / o_proj: half 0,
# down: half 1), then the embedding exchange's region. Each: [src rank][M][H] fp32
# (value, tag) pairs, sized for M = 8 (bf16 pair words use the first half of it).
AR_MMAX = 8


def peer_region_bytes(tp):
    return tp * AR_MMAX * H * 8


def peer_bytes(tp):
    return 3 * peer_region_bytes(tp)


CNT_REGION = dict(inp=0, out=1, qkv=2, o=3, gu=4, down=5, lm=6, attn=7)


PROFILE = {
    "addr": None
}  # debug: i64 [barriers][CUS][2] = (arrive, wait end) per CTA, 100 MHz


def build(cfg: Cfg):
    M, CUS, d = cfg.M, cfg.cus, cfg.d
    assert 1 <= M <= WAVES, "the norm prologues give each row one wave"
    assert cfg.ar_dtype in ("fp32", "bf16") and M <= AR_MMAX
    TP = cfg.tp
    AR = None
    if TP > 1:
        AR = dict(
            tp=TP,
            region=peer_region_bytes(TP),
            dtype=cfg.ar_dtype,
            local=cfg.ar_local,
        )
    SP = cfg.specs()
    S.NT = NT
    XQB = max(s.XR for s in SP.values()) * M
    XSF = 16 * max(s.XSR for s in SP.values())
    ATT_W = X.GQ * HD + HD  # floats per wave (q / p, then k_new v_new bf16 pairs)
    NTASK_G = M * d.nvh * 8
    NTASK_A = M * d.nkv * cfg.ncmax

    @fx.struct
    class GemvLds:
        xq: fx.Array[fx.Int32, (XQB + 3) // 4, 16]
        xs: fx.Array[fx.Float32, XSF, 16]

    @fx.struct
    class MixLds:
        aw: fx.Array[fx.Float32, WAVES * ATT_W, 16]

    @fx.union
    class StageLds:
        g: GemvLds
        x: MixLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, 16, 16]

    def tab64(r, i):
        v = D.ld(r, 2 * i, 2)
        return (fx.Int64(D.uniform(v[1])) << 32) | fx.Int64(fx.Uint32(D.uniform(v[0])))

    # FlyDSL's disk cache keys on the name, the source and scalar closure values, not
    # on cfg's fields: every field goes into the name, or two geometries share a binary
    name = f"qwen38_decode_m{M}_{zlib.crc32(repr(cfg).encode()):08x}"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kdec(tab: Int64, glob: Int64, nlayers: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = D.uniform(tid // 64)
        c = dict(
            tid=tid,
            bid=bid,
            lane=lane,
            wave=wave,
            gl=lane // 16,
            col=lane % 16,
            cus=CUS,
        )
        alloc = fx.SharedAllocator()
        red = alloc.allocate(Smem).peek().red.ptr
        st_lds = alloc.allocate(StageLds)
        gl_, xl_ = st_lds.g.peek(), st_lds.x.peek()
        xq, xs = gl_.xq.ptr, gl_.xs.ptr
        wl = xl_.aw.ptr + wave * ATT_W
        rg = D.rsrc(glob)
        rt = D.rsrc(tab)
        G = [tab64(rg, k) for k in range(NGLOB)]
        rf = D.rsrc(G[G_FLAGS])
        # every CTA reads the epoch before its first arrival; CTA 0 advances it after
        # the last barrier, which no CTA passes before all have read it
        epoch = D.uniform(D.ld(rf, (CUS + EPOCH_WORD_LINE) * D.LINE, cm=D.CM_DEV))
        hb0, hb1 = G[G_HB], G[G_HB] + fx.Int64(M * H * 4)
        hf0, _ = hf_ptrs(hb0, M)
        q0, q1 = G[G_SSQ], G[G_SSQ] + fx.Int64(M * (H // 16) * 4)
        cnt = lambda k: G[G_CNT] + fx.Int64(CNT_REGION[k] * CNT_STRIDE * 4)  # noqa: E731
        part = G[G_PART]
        if const_expr(TP > 1):
            # every rank's peer buffer, loaded once (a put's address must not wait on a
            # load of its own); this rank's from the table entry `rank`
            rp = D.rsrc(G[G_PEERS])
            rank = D.uniform(D.ld(rg, 2 * G_RANK))
            c["ar"] = dict(
                rank=rank,
                bases=[tab64(rp, p) for p in range(TP)],
                own=tab64(rp, rank),
                epoch=epoch,  # debug probes index barriers by tag - epoch - 1
            )

        def stamp(slot):
            t = D.memrealtime()
            if tid == 0:
                D.st(
                    fx.Vector.from_elements(
                        [fx.Int32(t & 0xFFFFFFFF), fx.Int32(t >> 32)], fx.Int32
                    ),
                    D.rsrc(fx.Int64(PROFILE["addr"])),
                    slot * 2,
                )

        def barrier_then(spec, tiles, scales, e):
            """Arrive, prefetch the GEMV's first units, wait."""
            w = None
            if const_expr(spec is not None):
                # opaque ids: keep the stage's address math inside the stage
                tid_s = D.opaque_v(tid)
                c.update(
                    tid=tid_s,
                    bid=D.opaque_s(bid),
                    lane=tid_s % 64,
                    wave=D.uniform(tid_s // 64),
                    gl=tid_s % 64 // 16,
                    col=tid_s % 16,
                )
                w = spec.setup(c, xq, xs, tiles, scales)
            if const_expr(PROFILE["addr"] is not None):
                slot = ((e - epoch - 1) * CUS + bid) * 2
                stamp(slot)
            apb = D.ARPROBE["bar"]
            if const_expr(apb is not None):
                t_arr = D.memrealtime()
                D.drain_cta()
                t_drn = D.memrealtime()
            D.grid_arrive(rf, e, tid, bid)
            buf = spec.first_loads(w) if spec is not None else None
            D.grid_wait(rf, e, tid, bid, CUS)
            if const_expr(PROFILE["addr"] is not None):
                stamp(slot + 1)
            if const_expr(apb is not None):
                t_rel = D.memrealtime()
                k = ((e - epoch - 1) * CUS + bid) * 4
                if tid == 0:
                    D.st_stamps(fx.Int64(apb), k, t_arr, t_drn)
                    D.st_stamps(fx.Int64(apb), k + 2, t_rel, fx.Int64(0))
            return w, buf

        # ---- embedding (+ zero the attention counters: their totals change with pos)
        S.embed_stage(c, M, G[G_TOK], G[G_EMBED], hb0, q0, hf0, epoch + 1)
        if bid == 0:
            for i in range_constexpr((M * d.nkv + NT - 1) // NT):
                if tid + NT * i < M * d.nkv:
                    D.st(fx.Int32(0), D.rsrc(cnt("attn")), tid + NT * i, D.CM_DEV)

        c0 = dict(c)
        for i in range(0, nlayers):
            # re-derive the thread ids inside the loop through opaque moves, so the
            # compiler cannot hoist every stage's per-lane address math out of the layer
            # loop (keeping it all live at once spills registers)
            tid_l = D.opaque_v(tid)
            bid_l = D.opaque_s(bid)
            lane_l, wave_l = tid_l % 64, D.uniform(tid_l // 64)
            c.update(
                tid=tid_l,
                bid=bid_l,
                lane=lane_l,
                wave=wave_l,
                gl=lane_l // 16,
                col=lane_l % 16,
            )
            gw_l = (
                wave_l * CUS + bid_l
            )  # task rank: consecutive tasks go to different CUs
            e0 = epoch + 1 + i * 5
            ri = i * NTAB
            T = functools.partial(lambda r, k: tab64(rt, r + k), ri)
            if i % 4 == 3:
                _attn_layer(
                    c,
                    SP,
                    T,
                    G,
                    e0,
                    xq,
                    xs,
                    red,
                    wl,
                    hb0,
                    hb1,
                    q0,
                    q1,
                    part,
                    cnt,
                    barrier_then,
                    gw_l,
                    NTASK_A,
                    cfg,
                )
            else:
                _gdn_layer(
                    c,
                    SP,
                    T,
                    G,
                    e0,
                    xq,
                    xs,
                    red,
                    wl,
                    hb0,
                    hb1,
                    q0,
                    q1,
                    part,
                    cnt,
                    barrier_then,
                    gw_l,
                    NTASK_G,
                    cfg,
                )
            _mlp(
                c,
                SP,
                T,
                e0 + 3,
                xq,
                xs,
                red,
                hb0,
                hb1,
                q0,
                q1,
                part,
                cnt,
                barrier_then,
                G,
            )
        c.clear()
        c.update(c0)
        e_end = epoch + 1 + nlayers * 5
        if const_expr(cfg.lm_head):
            # ---- final norm + LM head
            lm = SP["lm"]
            w, buf = barrier_then(lm, G[G_LM], G[G_LM], e_end)
            S.prologue_norm_bf16(c, lm, xq, red, hf0, q0, G[G_FNORM], EPS)
            lm.run(
                c,
                w,
                buf,
                part,
                cnt("lm"),
                lambda cg, tots: S.epi_raw(c, M, d.vocab, cg, tots[0], G[G_OUT]),
            )
        else:
            # ---- final norm -> bf16 hidden states
            barrier_then(None, None, None, e_end)
            S.final_norm_stage(c, M, hb0, q0, red, G[G_FNORM], G[G_OUT], EPS)
        if (bid == 0) & (tid == 0):
            D.st(e_end, rf, (CUS + EPOCH_WORD_LINE) * D.LINE, D.CM_DEV)
        if const_expr(PROFILE["addr"] is not None):
            stamp(((nlayers * 5 + 1) * CUS + bid) * 2)

    @flyc.jit
    def launch(tab: Int64, glob: Int64, nlayers: Int32, stream: fx.Stream = _STREAM):
        D.TRACE.clear()
        D.TRACE.update(d=d, ar=AR)
        kdec(tab, glob, nlayers).launch(grid=(CUS,), block=(NT,), stream=stream)

    return launch


def barriers_per_step(nlayers):
    return nlayers * 5 + 1


@D.traced
def _gdn_layer(
    c,
    SP,
    T,
    G,
    e0,
    xq,
    xs,
    red,
    wl,
    hb0,
    hb1,
    q0,
    q1,
    part,
    cnt,
    barrier_then,
    gw,
    ntask,
    cfg,
):
    M = c_M(SP)
    hf0, hf1 = hf_ptrs(hb0, M)
    inp, out = SP["inp"], SP["out"]
    y, y2, hssq = G[G_Y], G[G_Y2], G[G_HSSQ]
    # in_proj
    w, buf = barrier_then(inp, T(T_MIX_T), T(T_MIX_S), e0)
    S.prologue_norm(c, inp, xq, xs, red, hf0, q0, T(T_IN_LN), EPS)
    inp.run(
        c,
        w,
        buf,
        part,
        cnt("inp"),
        S.EpiConv(c, M, y, T(T_CONV), T(T_CONVW), T(T_IDX), cfg.conv_s0),
    )
    # GDN core
    barrier_then(None, None, None, e0 + 1)
    for t in range(gw, ntask, c["cus"] * WAVES):
        X.gdn_task(
            c,
            M,
            t,
            y,
            T(T_ALOG),
            T(T_DTB),
            T(T_NORMW),
            T(T_SSM),
            T(T_IDX),
            cfg.ssm_s0,
            y2,
            hssq,
            wl,
        )
    # out_proj (+ residual)
    w, buf = barrier_then(out, T(T_OUT_T), T(T_OUT_S), e0 + 2)
    S.prologue_scaled(c, out, xq, xs, y2, hssq, EPS, 128)
    out.run(c, w, buf, part, cnt("out"), S.EpiRes(c, M, hb0, hb1, q1, hf1, (e0 + 2, 0)))


@D.traced
def _attn_layer(
    c,
    SP,
    T,
    G,
    e0,
    xq,
    xs,
    red,
    wl,
    hb0,
    hb1,
    q0,
    q1,
    part,
    cnt,
    barrier_then,
    gw,
    ntask,
    cfg,
):
    M = c_M(SP)
    hf0, hf1 = hf_ptrs(hb0, M)
    qkv, o = SP["qkv"], SP["o"]
    y, y2 = G[G_Y], G[G_Y2]
    w, buf = barrier_then(qkv, T(T_MIX_T), T(T_MIX_S), e0)
    S.prologue_norm(c, qkv, xq, xs, red, hf0, q0, T(T_IN_LN), EPS)
    qkv.run(
        c,
        w,
        buf,
        part,
        cnt("qkv"),
        lambda cg, tots: S.epi_raw(c, M, D.TRACE["d"].qkv_n, cg, tots[0], y),
    )
    barrier_then(None, None, None, e0 + 1)
    for t in range(gw, ntask, c["cus"] * WAVES):
        X.attn_task(
            c,
            t,
            cfg,
            y,
            T(T_QN),
            T(T_KN),
            G[G_ROPE],
            T(T_KV),
            T(T_IDX),
            T(T_SLOT),
            G[G_POS],
            G[G_REC],
            cnt("attn"),
            y2,
            wl,
        )
    w, buf = barrier_then(o, T(T_OUT_T), T(T_OUT_S), e0 + 2)
    S.prologue_scaled(c, o, xq, xs, y2, y2, EPS, 0)
    o.run(c, w, buf, part, cnt("o"), S.EpiRes(c, M, hb0, hb1, q1, hf1, (e0 + 2, 0)))


@D.traced
def _mlp(c, SP, T, e0, xq, xs, red, hb0, hb1, q0, q1, part, cnt, barrier_then, G):
    M = c_M(SP)
    hf0, hf1 = hf_ptrs(hb0, M)
    gu, dn = SP["gu"], SP["down"]
    y = G[G_Y]
    w, buf = barrier_then(gu, T(T_GU_T), T(T_GU_S), e0)
    S.prologue_norm(c, gu, xq, xs, red, hf1, q1, T(T_POST_LN), EPS)
    gu.run(c, w, buf, part, cnt("gu"), lambda cg, tots: S.epi_act(c, M, cg, tots[0], y))
    w, buf = barrier_then(dn, T(T_DN_T), T(T_DN_S), e0 + 1)
    S.prologue_act(c, dn, xq, xs, y)
    dn.run(c, w, buf, part, cnt("down"), S.EpiRes(c, M, hb1, hb0, q0, hf0, (e0 + 1, 1)))


def hf_ptrs(hb0, M):
    """The bf16 copies of h0, h1 (after the two fp32 buffers, see Engine.hb)."""
    return hb0 + fx.Int64(2 * M * H * 4), hb0 + fx.Int64(2 * M * H * 4 + M * H * 2)


def c_M(SP):
    return SP["inp"].M
