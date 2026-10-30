# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.6-35B-A3B decode megakernel: one persistent launch per decode step.

Grid = CU count, 512 threads, all CTAs co-resident. Stages are separated by grid
barriers (dev.grid_arrive / grid_wait). Per layer (docs/DESIGN.md, "Per-layer stage
list"):

  GDN layer:  in_proj GEMV (fp8 qkvz + bf16 b/a) -> | GDN core tasks -> |
              out_proj (+h) -> |
  attn layer: qkv GEMV -> | attention tasks -> | o_proj (+h) -> |
  both:       router GEMV -> | routing + experts gate_up -> | experts down (+h) -> |
then the final norm and either the LM head GEMV (bf16) -> fp32 logits, or the normed
hidden states (bf16) for a serving engine's own LM head.

Residual stream: h0 at a layer start, h1 after the mixer (both fp32 [M][H], with
their 16-column sums of squares ssq0 / ssq1).

Caches use vLLM's layouts (Cfg holds their strides), as in the Qwen3.8 kernel:
  KV of an attention layer: bf16, K of (block, head, position) at element strides
    (kv_s0, kv_s1, kv_s2), its V kv_sv elements later; row m's block b is bt[m][b];
    or x-packed (ROCM_ATTN: kv_xpack, see mixer._attn_chunk).
  GDN layer: per state slot s, the conv state [3][CONV_DIM] bf16 (oldest input first)
    at conv + s conv_s0 bytes and the SSM state [NVH][DV][DK] fp32 at ssm + s ssm_s0
    bytes; row m's slot is sidx[m] (0: padding, not written).
  A row with slot[m] < 0 (padding) reads and writes no KV; a row with rowv[m] == 0
  selects no routed expert.
The barrier epoch lives on the device (word EPOCH_WORD of the flag buffer), so a
launch has no per-step host argument and can be captured in a graph.

Every compile-time setting is a Cfg field or a debug global. FlyDSL's disk cache keys on
mkcore.key.compile_key (the Cfg, the debug globals, the sources of mk/ and mkcore),
which the launcher holds; the kernel name holds a hash of that key.
"""

import dataclasses
import functools
import pathlib
from dataclasses import asdict, dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from . import mixer as X
from . import moe as MO
from . import stages as S
from .dims import (
    EPS,
    FI,
    HD,
    INP_N,
    NE,
    NKV,
    NL,
    NQ,
    NVH,
    QKV_N,
    QKVZ_N,
    VAL_DIM,
    VOCAB,
    H,
)
from .mkcore import dev as D
from .mkcore import key as K
from .mkcore import stagelib as SL
from .mkcore.dev import (
    _STREAM,
    CNT_STRIDE,
    EPOCH_WORD_LINE,
    LDS_MAX,
    NT,
    PROFILE,
    WAVES,
)
from .mkcore.gemv import Gemv

NTAB = 20  # pointer-table slots per layer
NGLOB = 20  # global pointer-table slots

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
    G_LOGIT,
    G_ACT,
    G_ROWV,
    G_SPARE,
) = range(NGLOB)
# Cfg.rt_geom 2: glob slot G_GEOM (after the slots above) holds the run-time cache
# geometry, the i32 words of mixer.GEO (Cfg.geom_slots); the attention stages load it
G_GEOM = NGLOB
NGEO = 1
# per-layer slots
(
    T_MIX_T,
    T_MIX_S,
    T_OUT_T,
    T_OUT_S,
    T_BA,
    T_RT,
    T_E1_T,
    T_E1_S,
    T_E2_T,
    T_E2_S,
    T_IN_LN,
    T_POST_LN,
) = range(12)
T_ALOG, T_DTB, T_CONVW, T_NORMW, T_CONV, T_SSM = range(12, 18)  # GDN layers
T_QN, T_KN, T_KV = range(12, 15)  # attention layers
# The layer's cache indices, per vLLM KV cache group: a GDN layer's state slots
# sidx[M]; an attention layer's block table bt[M][maxblocks] and slot mapping slot[M].
T_IDX, T_SLOT = 18, 19

STAGES_PER_LAYER = 6


@dataclass(frozen=True)
class Cfg:
    """One compiled kernel: M rows on `cus` CUs, the cache geometry it reads, and the
    GEMV K chunks (c_*)."""

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
    c_inp: int = 1
    c_ba: int = 8
    c_out: int = 1  # cta_red needs one chunk (2 with the global hand-off)
    c_qkv: int = 1
    c_o: int = 16  # one head per K chunk: the prologue merges one head (prologue_attn)
    c_router: int = 1
    c_e1: int = 4
    c_e2: int = 1
    c_lm: int = 1  # 8 -> 1: 3.36 -> 3.26 ms at M = 1 (fewer partial hand-offs)
    debug_route: bool = False  # store a hash of every CTA's routing tables (I5)
    fence: bool = False  # L2 write-back / invalidate at every grid barrier (dev.grid_*)
    chaos: bool = False  # random delays before arrivals and counter atomics (dev.CHAOS)
    # in_proj fp8, qkv, out_proj, router: CTA-local split-K, LDS reduction
    # (gemv.Gemv.cta; needs their K chunks = 1). False: the global last-arriver
    # hand-off (before 2026-10-29; docs/DESIGN.md, "Fixed stage cost")
    cta_red: bool = True
    # norm / scaled prologues only in CTAs that stream units of the GEMV (Gemv.busy):
    # in_proj b / a, out_proj, router
    skip_idle: bool = True
    rt_geom: int = 2  # 2: maxlen and maxblocks are run-time values (glob slot G_GEOM,
    # geom_slots), so one build serves every max_model_len; 0: compiled in

    def compiled(self):
        """The Cfg the kernel is built for: rt_geom zeroes its run-time fields
        (mixer.RT_FIELDS), so one build (one key, one name) serves every geometry that
        differs in them only."""
        assert self.rt_geom in (0, 2), self.rt_geom
        if not self.rt_geom:
            return self
        return dataclasses.replace(self, **{f: 0 for f in X.RT_FIELDS})

    def geom_slots(self):
        """The NGEO int64 glob slots from G_GEOM: the i32 words of mixer.GEO, two a
        slot (low word first)."""
        w = [getattr(self, n) for n in X.GEO]
        assert len(w) == 2 * NGEO and all(0 <= v < 2**31 for v in w), w
        return [w[2 * k] | w[2 * k + 1] << 32 for k in range(NGEO)]

    def chunks(self):
        return {k[2:]: v for k, v in asdict(self).items() if k.startswith("c_")}

    @property
    def cpb(self):
        """Attention chunks per KV block: a chunk never straddles two blocks."""
        return -(-self.block // X.CH)

    @property
    def ncmax(self):
        return self.maxblocks * self.cpb

    def specs(self):
        M, cus, ch = self.M, self.cus, self.chunks()

        def g(name, n, k, kind="fp8_b128", cta=False):
            return Gemv(
                name, n, k, M, cus, WAVES, ch[name], kind, cta=cta and self.cta_red
            ).check()

        return dict(
            inp=g("inp", QKVZ_N, H, cta=True),
            ba=g("ba", 2 * NVH, H, "bf16"),
            out=g("out", H, VAL_DIM, cta=True),
            qkv=g("qkv", QKV_N, H, cta=True),
            o=g("o", H, NQ * HD),
            router=g("router", MO.LGN, H, "bf16", cta=True),
            e1=MO.MoeGemv("e1", 2 * FI, H, M, cus, "gu", WAVES, ch["e1"]).check(),
            e2=MO.MoeGemv("e2", H, FI, M, cus, "down", WAVES, ch["e2"]).check(),
            lm=g("lm", VOCAB, H, "bf16"),
        )

    def part_floats(self):
        sp = self.specs()
        # in_proj's b/a GEMV keeps its partials after the fp8 GEMV's
        return max(
            max(s.part_floats() for s in sp.values()),
            sp["inp"].part_floats() + sp["ba"].part_floats(),
        )


CNT_REGION = dict(inp=0, ba=1, out=2, qkv=3, o=4, router=5, e1=6, e2=7, lm=8)


def build(cfg: Cfg):
    cfg = cfg.compiled()
    M, CUS = cfg.M, cfg.cus
    K.check_cfg(cfg)
    assert 1 <= M <= WAVES, "the norm prologues and the routing give each row one wave"
    SP = cfg.specs()
    D.configure(cfg)
    S.NT = NT
    NP = MO.npos_max(M)
    XQB = max(s.XR * (s.XROWS if isinstance(s, MO.MoeGemv) else M) for s in SP.values())
    XBB = SP["ba"].XR * M  # in_proj's bf16 x (b / a), beside the fp8 x
    XSF = 32 * max(s.XSR for s in SP.values())
    PRF = max([s.lds_floats() for s in SP.values() if isinstance(s, Gemv)] + [4])
    ATT_W = X.QW + HD  # floats per wave (q bf16, then k_new v_new bf16 pairs)
    NTASK_G = M * NVH * 8

    @fx.struct
    class GemvLds:
        xq: fx.Array[fx.Int32, (XQB + 3) // 4, 16]
        xb: fx.Array[fx.Int32, (XBB + 3) // 4, 16]
        xs: fx.Array[fx.Float32, XSF, 16]
        pr: fx.Array[fx.Float32, PRF, 16]  # cta GEMVs: the wave partials

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
        pick: fx.Array[fx.Int32, M * MO.TOPK, 16]
        rw: fx.Array[fx.Float32, M * MO.NSLOT, 16]
        eid: fx.Array[fx.Int32, NP + 3, 16]
        posof: fx.Array[fx.Int32, NE, 16]
        slot: fx.Array[fx.Int32, M * NP, 16]
        npos: fx.Array[fx.Int32, 4, 16]
        rmask: fx.Array[fx.Int32, M * 8, 16]  # routing: row m's 256-bit expert set

    lds = max(XQB + XBB + 4 * XSF + 4 * PRF, 4 * WAVES * ATT_W) + 4 * (
        16 + M * MO.TOPK + M * MO.NSLOT + NP + 3 + NE + M * NP + 4 + M * 8
    )
    assert lds <= LDS_MAX, f"STOP: I1 (LDS {lds} B > {LDS_MAX} B per CTA)"

    def tab64(r, i):
        v = D.ld(r, 2 * i, 2)
        return (fx.Int64(D.uniform(v[1])) << 32) | fx.Int64(fx.Uint32(D.uniform(v[0])))

    # the kernel name is not in FlyDSL's disk-cache key; `key` is (mkcore.key)
    key = K.compile_key(cfg, pathlib.Path(__file__).resolve().parent, PROFILE, D.PROBE)
    name = K.kernel_name("qwen36moe_decode", cfg, key)

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
        sm = alloc.allocate(Smem).peek()
        red = sm.red.ptr
        R = MO.Route(
            M,
            sm.pick.ptr,
            sm.rw.ptr,
            sm.eid.ptr,
            sm.posof.ptr,
            sm.slot.ptr,
            sm.npos.ptr,
            sm.rmask.ptr,
        )
        st_lds = alloc.allocate(StageLds)
        gl_, xl_ = st_lds.g.peek(), st_lds.x.peek()
        xq, xb, xs = gl_.xq.ptr, gl_.xb.ptr, gl_.xs.ptr
        c["pl"] = gl_.pr.ptr
        wl = xl_.aw.ptr + wave * ATT_W
        rg = D.rsrc(glob)
        rt = D.rsrc(tab)
        G = [tab64(rg, k) for k in range(NGLOB)]
        geo = X.Geo(cfg, rg, 2 * G_GEOM)
        rf = D.rsrc(G[G_FLAGS])
        # every CTA reads the epoch before its first arrival; CTA 0 advances it after
        # the last barrier, which no CTA passes before all have read it
        epoch = D.uniform(D.ld(rf, (CUS + EPOCH_WORD_LINE) * D.LINE, cm=D.CM_DEV))
        hb0, hb1 = G[G_HB], G[G_HB] + fx.Int64(M * H * 4)
        hf0, _ = hf_ptrs(hb0, M)
        q0, q1 = G[G_SSQ], G[G_SSQ] + fx.Int64(M * (H // 16) * 4)
        cnt = lambda k: G[G_CNT] + fx.Int64(CNT_REGION[k] * CNT_STRIDE * 4)  # noqa: E731
        part = G[G_PART]

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

        def barrier_then(spec, tiles, scales, e, pre=None):
            """Arrive, prefetch the GEMV's first units (or call pre: a stage without a
            GEMV loads its run-time geometry there), wait."""
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
            D.grid_arrive(rf, e, tid, bid, cfg.fence)
            pre_buf = pre() if pre is not None else None
            buf = spec.first_loads(w) if spec is not None else pre_buf
            D.grid_wait(rf, e, tid, bid, CUS, cfg.fence)
            if const_expr(PROFILE["addr"] is not None):
                stamp(slot + 1)
            return w, buf

        # ---- embedding
        S.embed_stage(c, M, G[G_TOK], G[G_EMBED], hb0, q0, hf0)

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
            e0 = epoch + 1 + i * STAGES_PER_LAYER
            ri = i * NTAB
            T = functools.partial(lambda r, k: tab64(rt, r + k), ri)
            env = dict(
                c=c,
                SP=SP,
                T=T,
                G=G,
                e0=e0,
                xq=xq,
                xb=xb,
                xs=xs,
                red=red,
                wl=wl,
                hb0=hb0,
                hb1=hb1,
                q0=q0,
                q1=q1,
                part=part,
                cnt=cnt,
                barrier_then=barrier_then,
                gw=gw_l,
                cfg=cfg,
                li=i,
            )
            if i % 4 == 3:
                _attn_layer(env, geo)
            else:
                _gdn_layer(env, NTASK_G)
            _moe(env, R)
        c.clear()
        c.update(c0)
        e_end = epoch + 1 + nlayers * STAGES_PER_LAYER
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
                lambda cg, tots: SL.epi_raw(c, M, VOCAB, cg, tots[0], G[G_OUT]),
            )
        else:
            # ---- final norm -> bf16 hidden states
            barrier_then(None, None, None, e_end)
            S.final_norm_stage(c, M, hb0, q0, red, G[G_FNORM], G[G_OUT], EPS)
        if (bid == 0) & (tid == 0):
            D.st(e_end, rf, (CUS + EPOCH_WORD_LINE) * D.LINE, D.CM_DEV)
        if const_expr(PROFILE["addr"] is not None):
            stamp(((nlayers * STAGES_PER_LAYER + 1) * CUS + bid) * 2)

    @flyc.jit
    def launch(tab: Int64, glob: Int64, nlayers: Int32, stream: fx.Stream = _STREAM):
        key  # noqa: B018  a str closure value: FlyDSL puts it in the disk-cache key
        kdec(tab, glob, nlayers).launch(grid=(CUS,), block=(NT,), stream=stream)

    return launch


def barriers_per_step(nlayers):
    return nlayers * STAGES_PER_LAYER + 1


def _gdn_layer(env, ntask):
    _gdn_layer_t(**env, ntask=ntask)


def _attn_layer(env, geo):
    _attn_layer_t(**env, geo=geo)


def _moe(env, R):
    _moe_t(**env, R=R)


@D.traced
def _gdn_layer_t(
    c,
    SP,
    T,
    G,
    e0,
    xq,
    xb,
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
    cfg,
    li,
    ntask,
):
    M = SP["inp"].M
    hf0, hf1 = hf_ptrs(hb0, M)
    inp, ba, out = SP["inp"], SP["ba"], SP["out"]
    y, y2, hssq = G[G_Y], G[G_Y2], G[G_HSSQ]
    # in_proj: fp8 [qkv | z] and bf16 [b | a] on the same normed input
    w, buf = barrier_then(inp, T(T_MIX_T), T(T_MIX_S), e0)
    S.prologue_norm(c, inp, xq, xs, red, hf0, q0, T(T_IN_LN), EPS)
    if const_expr(cfg.skip_idle):
        _if_busy(
            ba,
            c["bid"],
            lambda: S.prologue_norm_bf16(c, ba, xb, red, hf0, q0, T(T_IN_LN), EPS),
        )
    else:
        S.prologue_norm_bf16(c, ba, xb, red, hf0, q0, T(T_IN_LN), EPS)
    inp.run(
        c,
        w,
        buf,
        part,
        cnt("inp"),
        S.EpiConv(c, M, y, T(T_CONV), T(T_CONVW), T(T_IDX), cfg.conv_s0),
    )
    wb = ba.setup(c, xb, xs, T(T_BA), T(T_BA))
    ba.run(
        c,
        wb,
        ba.first_loads(wb),
        part + fx.Int64(inp.part_floats() * 4),
        cnt("ba"),
        lambda cg, tots: SL.epi_raw(c, M, INP_N, cg, tots[0], y, QKVZ_N),
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
    if const_expr(cfg.skip_idle):
        _if_busy(
            out, c["bid"], lambda: S.prologue_scaled(c, out, xq, xs, y2, hssq, EPS, 128)
        )
    else:
        S.prologue_scaled(c, out, xq, xs, y2, hssq, EPS, 128)
    out.run(c, w, buf, part, cnt("out"), S.EpiRes(c, M, hb0, hb1, q1, hf1))


@D.traced
def _attn_layer_t(
    c,
    SP,
    T,
    G,
    e0,
    xq,
    xb,
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
    cfg,
    li,
    geo,
):
    M = SP["inp"].M
    hf0, hf1 = hf_ptrs(hb0, M)
    qkv, o = SP["qkv"], SP["o"]
    y = G[G_Y]
    w, buf = barrier_then(qkv, T(T_MIX_T), T(T_MIX_S), e0)
    S.prologue_norm(c, qkv, xq, xs, red, hf0, q0, T(T_IN_LN), EPS)
    qkv.run(
        c,
        w,
        buf,
        part,
        cnt("qkv"),
        lambda cg, tots: SL.epi_raw(c, M, QKV_N, cg, tots[0], y),
    )
    _, g = barrier_then(None, None, None, e0 + 1, pre=geo.attn)
    ntask = M * NKV * g["ncmax"]
    for t in range(gw, ntask, c["cus"] * WAVES):
        X.attn_task(
            c,
            t,
            g,
            y,
            T(T_QN),
            T(T_KN),
            G[G_ROPE],
            T(T_KV),
            T(T_IDX),
            T(T_SLOT),
            G[G_POS],
            G[G_REC],
            wl,
        )
    w, buf = barrier_then(o, T(T_OUT_T), T(T_OUT_S), e0 + 2)
    X.prologue_attn(c, o, xq, xs, geo.attn(), G[G_REC], y, G[G_POS], T(T_SLOT))
    o.run(c, w, buf, part, cnt("o"), S.EpiRes(c, M, hb0, hb1, q1, hf1))


@D.traced
def _moe_t(
    c,
    SP,
    T,
    G,
    e0,
    xq,
    xb,
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
    cfg,
    li,
    R,
):
    M = SP["inp"].M
    hf0, hf1 = hf_ptrs(hb0, M)
    rtr, e1, e2 = SP["router"], SP["e1"], SP["e2"]
    # S4: router logits (+ the shared-expert gate) on bf16(norm(h1))
    w, buf = barrier_then(rtr, T(T_RT), T(T_RT), e0 + 3)
    if const_expr(cfg.skip_idle):
        _if_busy(
            rtr,
            c["bid"],
            lambda: S.prologue_norm_bf16(c, rtr, xq, red, hf1, q1, T(T_POST_LN), EPS),
        )
    else:
        S.prologue_norm_bf16(c, rtr, xq, red, hf1, q1, T(T_POST_LN), EPS)
    rtr.run(
        c,
        w,
        buf,
        part,
        cnt("router"),
        lambda cg, tots: SL.epi_raw(c, M, MO.LGN, cg, tots[0], G[G_LOGIT]),
    )
    # S5: routing (every CTA), then the experts' gate_up on fp8(norm(h1))
    barrier_then(None, None, None, e0 + 4)
    MO.routing(c, M, G[G_LOGIT], G[G_ROWV], R)
    if const_expr(cfg.debug_route):
        MO.route_hash(c, M, R, G[G_SPARE], li, c["cus"], NL)
    S.prologue_norm(c, e1, xq, xs, red, hf1, q1, T(T_POST_LN), EPS)
    w1 = e1.setup(c, xq, xs, T(T_E1_T), T(T_E1_S))
    e1.run(c, w1, R, part, cnt("e1"), MO.EpiAct(c, M, R, G[G_ACT], FI))
    # S6: the experts' down projection, the weighted slot sum and the residual
    barrier_then(None, None, None, e0 + 5)
    pos0 = e2.cta_pos0(c["bid"], R.n())
    MO.prologue_act_moe(c, e2, xq, xs, G[G_ACT], R, pos0)
    w2 = e2.setup(c, xq, xs, T(T_E2_T), T(T_E2_S))
    e2.run(c, w2, R, part, cnt("e2"), S.EpiRes(c, M, hb1, hb0, q0, hf0), pos0)


@D.traced
def _if_busy(spec, bid, fn):
    """fn() (a prologue) only in CTAs that stream units of GEMV `spec`."""
    if spec.busy(bid):
        fn()


def hf_ptrs(hb0, M):
    """The bf16 copies of h0, h1 (after the two fp32 buffers, see Engine.hb)."""
    return hb0 + fx.Int64(2 * M * H * 4), hb0 + fx.Int64(2 * M * H * 4 + M * H * 2)
