# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""The Gemma 4 26B-A4B decode megakernel: one persistent launch per decode step.

Structure from qwen3.6-35b-a3b-gfx942/mk/kernel.py at megakernels e2a6fb8 (grid, grid
barriers with the device epoch, opaque thread ids, pointer tables, Cfg and the kernel
name). Grid = CU count, 512 threads, all CTAs co-resident. Per layer (docs/DESIGN.md
"Stages and barriers"), `|` a grid barrier:

  | qkv GEMV (prologue: rebuild h, input_layernorm)
  | attention tasks (sliding window or full; K / V into vLLM's paged cache)
  | o_proj GEMV                          -> a (raw) + its 16-column sums of squares
  | router GEMV (prologue: h1 = h + post_attention_layernorm(a), router norm, bf16)
  | routing + expert gate_up (routed experts + the dense MLP as 3 pseudo-experts)
  | expert down                          -> e (weighted expert sum), d (dense MLP)
then | final norm (prologue: rebuild h) and either the LM head (bf16, soft-capped fp32
logits) or the bf16 normed hidden states for a serving engine's own LM head.
Barriers: 6 per layer + 1 = 181 per step.

Pointer table: row 0 holds the "layer -1" post-feedforward norms (ones) and layer
scalar (1.0), so layer 0's rebuild of h from the embedding needs no special case; layer
i is row i + 1, and the final norm rebuilds h from row nlayers.

Caches: vLLM TRITON_ATTN paged KV, one geometry (block size, element strides) for the
sliding layers and one for the full layers (Cfg); per layer the KV tensor, the block
table of its KV cache group and the slot mapping. A row with slot[m] < 0 (padding)
reads and writes no KV; a row with rowv[m] == 0 selects no routed expert.

Every compile-time setting is a Cfg field or a debug global, and the kernel name holds a
hash of all of them: FlyDSL's disk cache keys on the name, not on the fields.
"""

import functools
import zlib
from dataclasses import asdict, dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64

from . import attention as A
from . import dev as D
from . import moe as MO
from . import stages as S
from .dims import (
    EFF,
    EFFP,
    EMBED_SCALE,
    NE,
    NL,
    O_F,
    O_S,
    QKV_F,
    QKV_S,
    SOFTCAP,
    VOCAB,
    WINDOW,
    H,
)
from .gemv import Gemv

NTAB = 24  # pointer-table slots per row
NGLOB = 32  # global pointer-table slots
WAVES = 8
NT = 64 * WAVES
LDS_MAX = 65536

# global table slots
(
    G_EMBED,
    G_LM,
    G_FNORM,
    G_ROPE_S,
    G_ROPE_F,
    G_H,
    G_H1,
    G_RSTD1,
    G_A,
    G_SSQA,
    G_D,
    G_SSQD,
    G_E,
    G_SSQE,
    G_Y,
    G_O,
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
    G_SPARE2,
    G_XG,
    G_XGS,
    G_XQ,
    G_XQS,
) = range(NGLOB)
# per-row slots (row i + 1 = layer i)
(
    T_QKV_T,
    T_QKV_S,
    T_O_T,
    T_O_S,
    T_RT,
    T_E1_T,
    T_E1_S,
    T_E2_T,
    T_E2_S,
    T_IN_LN,
    T_PA_LN,
    T_PF_LN,
    T_PF2_LN,
    T_RSCALE,
    T_PES,
    T_QN,
    T_KN,
    T_KV,
    T_BT,
    T_SLOT,
    T_POST1,
    T_POST2,
    T_POSTF,
    T_LS,
) = range(NTAB)

_STREAM = fx.Stream(None)  # the launch's default: the current stream
CNT_STRIDE = 65536  # i32 counters per counter region
EPOCH_WORD_LINE = 8  # the epoch word is line cus + EPOCH_WORD_LINE of the flag buffer
STAGES_PER_LAYER = 6


@dataclass(frozen=True)
class Cfg:
    """One compiled kernel: M rows on `cus` CUs, the KV cache geometry of each layer type
    (block = positions per block; s0, s1, s2 = element strides of block, head, position;
    sv = elements from a K row to its V row), and the GEMV K chunks (c_*)."""

    M: int
    cus: int
    maxlen: int  # positions < maxlen (the RoPE tables' length)
    block_s: int
    maxblocks_s: int
    kv_s: tuple
    block_f: int
    maxblocks_f: int
    kv_f: tuple
    lm_head: bool = True  # False: write the normed hidden states instead of logits
    c_qkv: int = 1  # (the rebuild prologue fills all K)
    c_o: int = 4
    c_router: int = 1
    c_e1: int = 11  # 2816 = 11 x 256: 2 units per piece
    c_e2: int = 2  # 768 = 2 x 384: 3 units per piece
    c_lm: int = 1
    # router stage: only the CTAs with router units run its prologue; CTAs NR .. NR + M - 1
    # publish h1 and the quantized gate_up inputs (stages.publish_row), which the gate_up
    # prologue copies; else every CTA runs both whole-row prologues
    router_publish: bool = True
    # down stage: CTAs cus - M + m wait for all expert-down groups (a device counter), then
    # rebuild row m of h and publish the next layer's quantized qkv input
    # (stages.combine_row), which the qkv prologue copies (layers > 0); else every CTA
    # rebuilds all rows of h in the qkv prologue
    combine_rows: bool = True
    debug_route: bool = False  # store a hash of every CTA's routing tables (I5)
    fence: bool = False  # L2 write-back / invalidate at every grid barrier (dev.grid_*)

    def chunks(self):
        return {k[2:]: v for k, v in asdict(self).items() if k.startswith("c_")}

    def geo(self, full):
        if full:
            nc = -(-self.maxlen // A.CH)
            return A.Geo(True, nc, self.maxblocks_f, self.block_f, *self.kv_f).check()
        return A.Geo(
            False, WINDOW // A.CH + 1, self.maxblocks_s, self.block_s, *self.kv_s
        ).check()

    def specs(self):
        M, cus, ch = self.M, self.cus, self.chunks()

        def g(name, n, k, c, kind="fp8"):
            return Gemv(name, n, k, M, cus, WAVES, c, kind).check()

        return dict(
            qkv_s=g("qkv_s", QKV_S, H, ch["qkv"]),
            qkv_f=g("qkv_f", QKV_F, H, ch["qkv"]),
            o_s=g("o_s", H, O_S, ch["o"]),
            o_f=g("o_f", H, O_F, ch["o"]),
            router=g("router", MO.LGN, H, ch["router"], "bf16"),
            e1=MO.MoeGemv("e1", 2 * EFF, H, M, cus, "gu", WAVES, ch["e1"]).check(),
            e2=MO.MoeGemv("e2", H, EFFP, M, cus, "down", WAVES, ch["e2"]).check(),
            lm=g("lm", VOCAB, H, ch["lm"], "bf16"),
        )

    def part_floats(self):
        return max(s.part_floats() for s in self.specs().values())


CNT_REGION = dict(qkv=0, o=1, router=2, e1=3, e2=4, lm=5, attn=6, done=7)


PROFILE = {
    "addr": None
}  # debug: i64 [barriers][CUS][2] = (arrive, wait end) per CTA, 100 MHz


def kernel_name(cfg: Cfg) -> str:
    """Every compile-time setting goes into the name (FlyDSL's disk cache keys on it):
    the Cfg fields and the debug globals (PROFILE and PROBE addresses are baked in)."""
    key = repr((cfg, PROFILE, D.PROBE))
    return f"gemma4_decode_m{cfg.M}_{zlib.crc32(key.encode()):08x}"


def lds_bytes(cfg: Cfg):
    M = cfg.M
    SP = cfg.specs()
    XQB = max(s.XR * (s.XROWS if isinstance(s, MO.MoeGemv) else M) for s in SP.values())
    XSF = 32 * max(s.XSR for s in SP.values())
    AW = max(cfg.geo(f).WLF for f in (False, True))
    NP = MO.npos_max(M)
    fixed = 4 * (
        S.RED_FLOATS + M * MO.TOPK + M * MO.NSLOT + NP + 3 + NE + M * NP + 4 + M * 8
    )
    return XQB, XSF, AW, NP, max(XQB + 4 * XSF, 4 * WAVES * AW) + fixed


def build(cfg: Cfg):
    M, CUS = cfg.M, cfg.cus
    assert 1 <= M <= WAVES, "the routing and the rstd reductions give each row one wave"
    SP = cfg.specs()
    GEO = {False: cfg.geo(False), True: cfg.geo(True)}
    D.FENCE["on"] = cfg.fence
    S.NT = NT
    XQB, XSF, AW, NP, lds = lds_bytes(cfg)
    assert lds <= LDS_MAX, f"STOP: I1 (LDS {lds} B > {LDS_MAX} B per CTA)"

    @fx.struct
    class GemvLds:
        xq: fx.Array[fx.Int32, (XQB + 3) // 4, 16]
        xs: fx.Array[fx.Float32, XSF, 16]

    @fx.struct
    class MixLds:
        aw: fx.Array[fx.Float32, WAVES * AW, 16]

    @fx.union
    class StageLds:
        g: GemvLds
        x: MixLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, S.RED_FLOATS, 16]
        pick: fx.Array[fx.Int32, M * MO.TOPK, 16]
        rw: fx.Array[fx.Float32, M * MO.NSLOT, 16]
        eid: fx.Array[fx.Int32, NP + 3, 16]
        posof: fx.Array[fx.Int32, NE, 16]
        slot: fx.Array[fx.Int32, M * NP, 16]
        npos: fx.Array[fx.Int32, 4, 16]
        rmask: fx.Array[fx.Int32, M * 8, 16]  # routing: row m's expert set (bits)

    def tab64(r, i):
        v = D.ld(r, 2 * i, 2)
        return (fx.Int64(D.uniform(v[1])) << 32) | fx.Int64(fx.Uint32(D.uniform(v[0])))

    name = kernel_name(cfg)

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
        xq, xs = gl_.xq.ptr, gl_.xs.ptr
        rg = D.rsrc(glob)
        rt = D.rsrc(tab)
        G = [tab64(rg, k) for k in range(NGLOB)]
        rf = D.rsrc(G[G_FLAGS])
        # every CTA reads the epoch before its first arrival; CTA 0 advances it after
        # the last barrier, which no CTA passes before all have read it
        epoch = D.uniform(D.ld(rf, (CUS + EPOCH_WORD_LINE) * D.LINE, cm=D.CM_DEV))
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
            D.grid_arrive(rf, e, tid, bid, cfg.fence)
            buf = spec.first_loads(w) if spec is not None else None
            D.grid_wait(rf, e, tid, bid, CUS, cfg.fence)
            if const_expr(PROFILE["addr"] is not None):
                stamp(slot + 1)
            return w, buf

        # ---- embedding (layer 0's rebuild inputs; row 0 of the table: ones, 1.0)
        S.embed_stage(
            c,
            M,
            G[G_TOK],
            G[G_EMBED],
            EMBED_SCALE,
            G[G_H1],
            G[G_D],
            G[G_E],
            G[G_SSQD],
            G[G_SSQE],
        )

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
            T = functools.partial(lambda r, k: tab64(rt, r + k), (i + 1) * NTAB)
            Tp = functools.partial(lambda r, k: tab64(rt, r + k), i * NTAB)
            Tn = functools.partial(lambda r, k: tab64(rt, r + k), (i + 2) * NTAB)
            env = dict(
                c=c,
                SP=SP,
                T=T,
                Tp=Tp,
                Tn=Tn,
                G=G,
                e0=e0,
                xq=xq,
                xs=xs,
                red=red,
                wl=xl_.aw.ptr,
                part=part,
                cnt=cnt,
                barrier_then=barrier_then,
                gw=gw_l,
                cfg=cfg,
                li=i,
            )
            if i % 6 == 5:
                _attn_layer(env, GEO[True], "f")
            else:
                _attn_layer(env, GEO[False], "s")
            _moe(env, R, nlayers)
        c.clear()
        c.update(c0)
        e_end = epoch + 1 + nlayers * STAGES_PER_LAYER
        Tl = functools.partial(lambda r, k: tab64(rt, r + k), nlayers * NTAB)
        lm = SP["lm"]
        if const_expr(cfg.lm_head):
            w, buf = barrier_then(lm, G[G_LM], G[G_LM], e_end)
        else:
            barrier_then(None, None, None, e_end)
        hh, k0, live = S.rebuild_h(
            c,
            M,
            red,
            G[G_H1],
            G[G_D],
            G[G_E],
            G[G_SSQD],
            G[G_SSQE],
            Tl(T_POST1),
            Tl(T_POST2),
            Tl(T_POSTF),
            Tl(T_LS),
            G[G_H],
        )
        S.prologue_final(
            c, M, xq, lm.XR, red, hh, k0, live, G[G_FNORM], G[G_OUT], cfg.lm_head
        )
        if const_expr(cfg.lm_head):
            lm.run(
                c,
                w,
                buf,
                part,
                cnt("lm"),
                lambda cg, tots: S.epi_softcap(
                    c, M, VOCAB, cg, tots[0], G[G_OUT], SOFTCAP
                ),
            )
        if (bid == 0) & (tid == 0):
            D.st(e_end, rf, (CUS + EPOCH_WORD_LINE) * D.LINE, D.CM_DEV)
        if const_expr(PROFILE["addr"] is not None):
            stamp(((nlayers * STAGES_PER_LAYER + 1) * CUS + bid) * 2)

    @flyc.jit
    def launch(tab: Int64, glob: Int64, nlayers: Int32, stream: fx.Stream = _STREAM):
        kdec(tab, glob, nlayers).launch(grid=(CUS,), block=(NT,), stream=stream)

    return launch


def barriers_per_step(nlayers):
    return nlayers * STAGES_PER_LAYER + 1


def _attn_layer(env, geo, kind):
    _attn_layer_t(**env, geo=geo, kind=kind)


def _moe(env, R, nlayers):
    _moe_t(**env, R=R, nlayers=nlayers)


@D.traced
def _attn_layer_t(
    c,
    SP,
    T,
    Tp,
    Tn,
    G,
    e0,
    xq,
    xs,
    red,
    wl,
    part,
    cnt,
    barrier_then,
    gw,
    cfg,
    li,
    geo,
    kind,
):
    qkv, o = SP["qkv_" + kind], SP["o_" + kind]
    M = qkv.M
    rope = G[G_ROPE_F] if geo.full else G[G_ROPE_S]
    # qkv: rebuild h (the previous layer's raw outputs), input_layernorm, fp8
    w, buf = barrier_then(qkv, T(T_QKV_T), T(T_QKV_S), e0)
    if const_expr(cfg.combine_rows):
        if (c["bid"] == 0) & (c["tid"] == 0):
            # the previous layer's combine CTAs have passed the done counter
            D.st(fx.Int32(0), D.rsrc(cnt("done")), 0, D.CM_DEV)
        if li == 0:
            _qkv_rebuild(c, M, qkv, xq, xs, red, G, T, Tp)
        else:  # the previous layer's combine CTAs published x (and h)
            S.prologue_gu_copy(c, qkv, xq, xs, G[G_XQ], G[G_XQS], M)
    else:
        _qkv_rebuild(c, M, qkv, xq, xs, red, G, T, Tp)
    qkv.run(
        c,
        w,
        buf,
        part,
        cnt("qkv"),
        lambda cg, tots: S.epi_raw(c, M, qkv.N, cg, tots[0], G[G_Y]),
    )
    # attention tasks
    barrier_then(None, None, None, e0 + 1)
    wlw = wl + c["wave"] * geo.WLF
    for t in range(gw, geo.ntask(M), c["cus"] * WAVES):
        A.attn_task(
            c,
            geo,
            t,
            G[G_Y],
            T(T_QN),
            T(T_KN),
            rope,
            T(T_KV),
            T(T_BT),
            T(T_SLOT),
            G[G_POS],
            G[G_REC],
            cnt("attn"),
            G[G_O],
            wlw,
        )
    # o_proj -> a (raw) and its sums of squares
    w, buf = barrier_then(o, T(T_O_T), T(T_O_S), e0 + 2)
    S.prologue_scaled(c, o, xq, xs, G[G_O])
    o.run(c, w, buf, part, cnt("o"), S.EpiSsq(c, M, G[G_A], G[G_SSQA]))


@D.traced
def _qkv_rebuild(c, M, qkv, xq, xs, red, G, T, Tp):
    hh, k0, live = S.rebuild_h(
        c,
        M,
        red,
        G[G_H1],
        G[G_D],
        G[G_E],
        G[G_SSQD],
        G[G_SSQE],
        Tp(T_POST1),
        Tp(T_POST2),
        Tp(T_POSTF),
        Tp(T_LS),
        G[G_H],
        qkv.name,
    )
    S.prologue_qkv(c, qkv, xq, xs, red, hh, k0, live, T(T_IN_LN))


@D.traced
def _moe_t(
    c,
    SP,
    T,
    Tp,
    Tn,
    G,
    e0,
    xq,
    xs,
    red,
    wl,
    part,
    cnt,
    barrier_then,
    gw,
    cfg,
    li,
    R,
    nlayers,
):
    rtr, e1, e2 = SP["router"], SP["e1"], SP["e2"]
    M = rtr.M
    # debug_route: every layer's router logits stay ([NL][M][LGN]); else one buffer
    logit = G[G_LOGIT]
    if const_expr(cfg.debug_route):
        logit = logit + fx.Int64(li) * (M * MO.LGN * 4)
    # router: h1 = h + post_attention_layernorm(a); x = bf16(router norm)
    w, buf = barrier_then(rtr, T(T_RT), T(T_RT), e0 + 3)
    if const_expr(cfg.router_publish):
        NR = -(-min(rtr.U, rtr.NW) // WAVES)  # CTAs with router units (a "small" GEMV)
        assert rtr.small and cfg.cus >= NR + M
        bid = c["bid"]
        if bid < NR:
            S.prologue_router(
                c,
                rtr,
                xq,
                red,
                G[G_H],
                G[G_A],
                G[G_SSQA],
                T(T_PA_LN),
                T(T_RSCALE),
                G[G_H1],
                G[G_RSTD1],
                store=False,
            )
        elif bid < NR + M:
            S.publish_row(
                c,
                M,
                bid - NR,
                red,
                G[G_H],
                G[G_A],
                G[G_SSQA],
                T(T_PA_LN),
                T(T_PF2_LN),
                T(T_PF_LN),
                G[G_H1],
                G[G_RSTD1],
                G[G_XG],
                G[G_XGS],
            )
    else:
        S.prologue_router(
            c,
            rtr,
            xq,
            red,
            G[G_H],
            G[G_A],
            G[G_SSQA],
            T(T_PA_LN),
            T(T_RSCALE),
            G[G_H1],
            G[G_RSTD1],
        )
    rtr.run(
        c,
        w,
        buf,
        part,
        cnt("router"),
        lambda cg, tots: S.epi_raw(c, M, MO.LGN, cg, tots[0], logit),
    )
    # routing (every CTA), then gate_up of the routed experts and the dense MLP
    barrier_then(None, None, None, e0 + 4)
    MO.routing(c, M, logit, G[G_ROWV], T(T_PES), R)
    if const_expr(cfg.debug_route):
        MO.route_hash(c, M, R, G[G_SPARE], li, c["cus"], NL)
    if const_expr(cfg.router_publish):
        S.prologue_gu_copy(c, e1, xq, xs, G[G_XG], G[G_XGS])
    else:
        S.prologue_gu(c, e1, xq, xs, red, G[G_H1], G[G_RSTD1], T(T_PF2_LN), T(T_PF_LN))
    w1 = e1.setup(c, xq, xs, T(T_E1_T), T(T_E1_S))
    e1.run(c, w1, R, part, cnt("e1"), MO.EpiAct(c, M, R, G[G_ACT], EFFP))
    # down: weighted expert sum e and dense sum d, raw, with their sums of squares
    barrier_then(None, None, None, e0 + 5)
    pos0 = e2.cta_pos0(c["bid"], R.n())
    MO.prologue_act_moe(c, e2, xq, xs, G[G_ACT], R, pos0)
    w2 = e2.setup(c, xq, xs, T(T_E2_T), T(T_E2_S))
    if const_expr(cfg.combine_rows):
        more = li + 1 < nlayers  # the last layer's h is rebuilt by the final stage
        e2.run(
            c,
            w2,
            R,
            part,
            cnt("e2"),
            S.EpiDE(c, M, G[G_E], G[G_SSQE], G[G_D], G[G_SSQD], cnt("done"), more),
            pos0,
        )
        m = c["bid"] - (cfg.cus - M)
        if more & (m >= 0):
            if c["tid"] == 0:
                D.poll_ge(D.rsrc(cnt("done")), 0, fx.Int32(e2.G))
            gpu.barrier()  # every expert-down group has stored d, e and their sums
            S.combine_row(
                c,
                m,
                red,
                G[G_H1],
                G[G_D],
                G[G_E],
                G[G_SSQD],
                G[G_SSQE],
                T(T_POST1),
                T(T_POST2),
                T(T_POSTF),
                T(T_LS),
                Tn(T_IN_LN),
                G[G_H],
                G[G_XQ],
                G[G_XQS],
            )
    else:
        e2.run(
            c,
            w2,
            R,
            part,
            cnt("e2"),
            S.EpiDE(c, M, G[G_E], G[G_SSQE], G[G_D], G[G_SSQD]),
            pos0,
        )
