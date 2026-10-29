# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side helpers of the megakernel (FlyDSL, gfx942).

Memory scopes (gfx94x CPol): CM_DEV = sc1 makes a store/load coherent across the
XCDs (device scope). Every hand-off between CTAs uses CM_DEV on both sides.
Spin loops contain s_sleep: a side effect that keeps LLVM from deleting the loop.
"""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

CM_DEV = 16
CM_NT = 2
LINE = 32  # i32 per 128 B cache line

traced = ASTRewriter.transform


def rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def gptr(addr):
    return llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), fx.Int64(addr).ir_value()
    ).result


def gld32(addr):
    """The i32 at a per-lane 64-bit global address (4-aligned): for data whose rows
    differ across a wave and lie more than 4 GB apart, where a buffer resource would
    have to be divergent."""
    return fx.Int32(llvm.load(T.i32, gptr(addr), alignment=4))


def gst_bf16(v, addr):
    """Store f32 v as bf16 at a per-lane 64-bit global address."""
    llvm.store(fx.Float32(v).to(fx.BFloat16).ir_value(), gptr(addr), alignment=2)


def ld(r, i, n=1, dtype=None, cm=0):
    """N (1, 2, 4) elements of `dtype` (default i32) at element offset i."""
    dt = T.i32 if dtype is None else dtype
    v = bo.buffer_load(r, i, vec_width=n, dtype=dt, cache_modifier=cm)
    if n == 1:
        return fx.Int32(v) if dtype is None else fx.Float32(v)
    return fx.Vector(v)


def ldf(r, i, n=1, cm=0):
    return ld(r, i, n, T.f32, cm)


def st(v, r, i, cm=0):
    bo.buffer_store(v, r, i, cache_modifier=cm)


def opaque_s(v):
    """V (wave-uniform i32) through an asm s_mov with side effects: the compiler cannot
    hoist what depends on the result out of the enclosing loop."""
    return fx.Int32(
        llvm.InlineAsmOp(
            T.i32,
            [fx.Int32(v).ir_value()],
            "s_mov_b32 $0, $1",
            "=s,s",
            has_side_effects=True,
        ).result
    )


def opaque_v(v):
    """Per-lane i32 version of opaque_s (v_mov)."""
    return fx.Int32(
        llvm.InlineAsmOp(
            T.i32,
            [fx.Int32(v).ir_value()],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        ).result
    )


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def gather(v, src_lane):
    """Lane src_lane's i32 value (ds_bpermute)."""
    return fx.Int32(
        rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v).ir_value())
    )


def gather_f(v, src_lane):
    return gather(fx.Float32(v).bitcast(fx.Int32), src_lane).bitcast(fx.Float32)


# Butterfly steps inside a 16-lane row as DPP (no LDS round trip). Steps 1 and 2 are
# exact xor partners; 4 and 8 use the row mirrors, which pair the right lanes once the
# smaller steps are done (so only a prefix 1, 2, 4, 8 of a full reduction uses DPP).
_DPP = (
    (1, "quad_perm:[1,0,3,2]"),
    (2, "quad_perm:[2,3,0,1]"),
    (4, "row_half_mirror"),
    (8, "row_mirror"),
)


def _dpp(op, v, ctrl):
    # s_nop 1: a DPP read of a VGPR that a VALU op just wrote needs 2 wait states
    return fx.Float32(
        llvm.InlineAsmOp(
            T.f32,
            [fx.Float32(v).ir_value()],
            f"s_nop 1\n{op}_dpp $0, $1, $1 {ctrl} row_mask:0xf bank_mask:0xf",
            "=v,v",
        ).result
    )


def _reduce(v, lane, offs, op, comb):
    offs = list(offs)
    for o, ctrl in _DPP:
        if not offs or offs[0] != o:
            break
        v = _dpp(op, v, ctrl)
        offs.pop(0)
    for o in offs:
        v = comb(v, gather_f(v, lane ^ o))
    return v


def xor_sum(v, lane, offs):
    """Sum over the lanes lane ^ (any subset sum of offs); every lane gets it."""
    return _reduce(v, lane, offs, "v_add_f32", lambda a, b: a + b)


def xor_max(v, lane, offs):
    return _reduce(v, lane, offs, "v_max_f32", fx.max)


def _dpp_rows(op, v, ctrl, rows):
    """DPP op on the lanes of the rows in mask `rows` only; the other lanes keep v (the
    output is tied to the input)."""
    return fx.Float32(
        llvm.InlineAsmOp(
            T.f32,
            [fx.Float32(v).ir_value()],
            f"s_nop 1\n{op}_dpp $0, $1, $1 {ctrl} row_mask:{rows} bank_mask:0xf",
            "=v,0",
        ).result
    )


def _wave_uniform(v, op):
    """Reduce v over the 64 lanes with DPP only (gfx9): the 16-lane butterfly, then
    row_bcast:15 and row_bcast:31 carry the row results up to lane 63, which is read
    as a wave-uniform scalar. Fixed order: ((r0 . r1) . (r2 . r3)) per lane slot."""
    for _, ctrl in _DPP:
        v = _dpp(op, v, ctrl)
    v = _dpp_rows(op, v, "row_bcast:15", "0xa")
    v = _dpp_rows(op, v, "row_bcast:31", "0xc")
    w = llvm.call_intrinsic(
        T.i32,
        "llvm.amdgcn.readlane.i32",
        [fx.Float32(v).bitcast(fx.Int32).ir_value(), fx.Int32(63).ir_value()],
        [],
        [],
    )
    return fx.Int32(w).bitcast(fx.Float32)


def wave_max(v):
    """Max of v over the wave, as a wave-uniform value (no LDS round trip)."""
    return _wave_uniform(v, "v_max_f32")


def wave_sum(v):
    """Sum of v over the wave in a fixed order, as a wave-uniform value."""
    return _wave_uniform(v, "v_add_f32")


def absf(v):
    return fx.Float32(fmath.absf(v))


def fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def fma4(a, b, c):
    return fx.Vector.from_elements(
        [fmath.fma(a[r], b[r], c[r]) for r in range(4)], fx.Float32
    )


def rsq(x):
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32, "llvm.amdgcn.rsq.f32", [fx.Float32(x).ir_value()], [], []
        )
    )


def exp2(x):
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32, "llvm.amdgcn.exp2.f32", [fx.Float32(x).ir_value()], [], []
        )
    )


def log2(x):
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32, "llvm.amdgcn.log.f32", [fx.Float32(x).ir_value()], [], []
        )
    )


def logf1p_exp(x):
    """log(1 + exp(x)) (softplus below its linear threshold)."""
    return log2(fx.Float32(1.0) + expf(x)) * 0.6931471805599453


def expf(x):
    return exp2(fx.Float32(x) * 1.4426950408889634)


def sigmoid(x):
    return fx.Float32(1.0) / (fx.Float32(1.0) + expf(-fx.Float32(x)))


def silu(x):
    return fx.Float32(x) * sigmoid(x)


def bf_lo(w):
    return (fx.Int32(w) << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (fx.Int32(w) & fx.Int32(-65536)).bitcast(fx.Float32)


def pow2_scale(amax):
    """(qinv, scale): powers of two with amax * qinv in [120, 240] (e4m3fnuz max 240).
    Mirrors reference.pow2_scale. amax must be > 0."""
    bits = fx.Float32(amax).bitcast(fx.Int32)
    e = (bits >> 23) & 0xFF
    e = e + ((bits & 0x7FFFFF) > 0x700000).select(fx.Int32(1), fx.Int32(0))
    return ((261 - e) << 23).bitcast(fx.Float32), ((e - 7) << 23).bitcast(fx.Float32)


def fp8_pack4(a, b, c, d):
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


def quant8(f, qi):
    """8 f32 x qi -> 2 dwords of e4m3fnuz."""
    return (
        fp8_pack4(f[0] * qi, f[1] * qi, f[2] * qi, f[3] * qi),
        fp8_pack4(f[4] * qi, f[5] * qi, f[6] * qi, f[7] * qi),
    )


def memrealtime():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


@traced
def poll_ge(r, slot, val):
    """Spin until word `slot` (device scope) >= val, in wrap-around order (epochs
    are i32 counters that overflow after about 6.7 M steps)."""
    v = ld(r, slot, cm=CM_DEV)
    while (v - val) < 0:
        rocdl.s_sleep(1)
        v = ld(r, slot, cm=CM_DEV)


def _asm(text):
    llvm.InlineAsmOp(None, [], text, "", has_side_effects=True)


@traced
def grid_arrive(rf, epoch, tid, bid, fence=False):
    """Barrier arrival: drain this CTA's memory ops (its stores become visible), then
    store epoch to the CTA's flag line. Issue prefetch loads after this, not before:
    s_waitcnt vmcnt(0) also waits for loads (measured: prefetching before the arrival
    is slower). fence: also write back this XCD's dirty L2 lines first (agent-scope
    release, buffer_wbl2 sc1)."""
    if const_expr(CHAOS["on"]):
        chaos_sleep(bid)
    if const_expr(fence):
        _asm("buffer_wbl2 sc1")
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        st(epoch, rf, bid * LINE, CM_DEV)


@traced
def grid_wait(rf, epoch, tid, bid, cus, fence=False):
    """Barrier completion. Leader x < 8 waits for the CTAs b % 8 == x (its XCD under
    round-robin dispatch) and stores epoch to line cus + x; every CTA waits for the
    8 leader lines. Epochs only grow. fence: then invalidate this XCD's non-local L2
    lines and the CU's L1 (agent-scope acquire, buffer_inv sc1). (One-hop variants,
    device-scope arrival counters and a dense flag array polled by every CTA, were
    slower: docs/DESIGN.md, "Fixed stage cost".)"""
    if bid < 8:
        nx = (cus - bid + 7) // 8
        if tid < nx:
            poll_ge(rf, (bid + 8 * tid) * LINE, epoch)
        gpu.barrier()
        if tid == 0:
            st(epoch, rf, (cus + bid) * LINE, CM_DEV)
    if tid < 8:
        poll_ge(rf, (cus + tid) * LINE, epoch)
    gpu.barrier()
    if const_expr(fence):
        _asm("buffer_inv sc1")
        rocdl.s_waitcnt(vmcnt=0)


FENCE = {"on": False}  # debug (Cfg.fence): agent-scope release / acquire around atomics
CHAOS = {
    "on": False
}  # debug (Cfg.chaos): random 0..7 us delays before arrivals and atomics


@traced
def chaos_sleep(salt):
    """Debug: sleep 0..7 x s_sleep(31) (about 1 us each), drawn from the clock and
    `salt`. Changes the arrival order at barriers and counters from run to run; the
    results must not change (P1, P2)."""
    x = (fx.Int32(memrealtime()) ^ (fx.Int32(salt) * fx.Int32(-1640531535))) * fx.Int32(
        -2048144789
    )
    n = (x >> 28) & 7
    while n > 0:
        rocdl.s_sleep(31)
        n = n - 1


def atomic_add_dev(addr, v):
    """Device-scope relaxed atomic add on the i32 at `addr`; returns the old value.

    Relaxed on purpose: an acquire/release at agent scope makes LLVM write back
    (buffer_wbl2) and invalidate (buffer_inv) the whole L2, which stalls the weight
    stream for tens of us. Ordering comes from the protocol instead: the payload is
    stored with sc1 and drained (s_waitcnt vmcnt(0)) before the atomic, and readers
    load it with sc1. FENCE["on"] adds the explicit release / acquire (debug)."""
    if const_expr(CHAOS["on"]):
        chaos_sleep(0)
    if FENCE["on"]:
        _asm("buffer_wbl2 sc1")
        rocdl.s_waitcnt(vmcnt=0)
    old = fx.Int32(
        fx.atomic_add(
            gptr(addr),
            fx.Int32(v),
            syncscope=fx.rocdl.SyncScope.Agent,
            ordering=fx.AtomicOrdering.Monotonic,
        )
    )
    if FENCE["on"]:
        rocdl.s_waitcnt(vmcnt=0)
        _asm("buffer_inv sc1")
    return old


PROBE = {"addr": None}  # debug: global address of an i64 array [CTA * 8 + wave][16]


def probe(c, slot):
    """Debug: lane 0 of each wave stores the 100 MHz clock to probe slot `slot`."""
    if PROBE["addr"] is None or c is None or not PROBE.get("on", True):
        return
    _probe(c["lane"], c["bid"] * PROBE.get("waves", 8) + c["wave"], slot, PROBE["addr"])


@traced
def _probe(lane, wid, slot, addr):
    t = memrealtime()
    if lane == 0:
        st(
            fx.Vector.from_elements(
                [fx.Int32(t & 0xFFFFFFFF), fx.Int32(t >> 32)], fx.Int32
            ),
            rsrc(addr),
            (wid * 16 + slot) * 2,
        )
