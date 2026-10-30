# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 trace-time state and TP debug helpers (were in mk/dev.py; common/DESIGN.md
R-state: model state never lives in mkcore). They stay with Qwen3.8 TP until a second
model uses TP (D10)."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import gpu, rocdl

from .mkcore.dev import rsrc, st

# Trace-time constants of the kernel being traced (kernel.build's launch sets them
# before its body traces): "d" the rank's Dims, "ar" the all-reduce config or None.
# Python values cannot ride in the stage dict c, which the layer loop carries as IR.
TRACE = {}

# debug (TP > 1): all-reduce timing split, docs/DESIGN-TP2.md section 11. "bar": i64
# [barrier][CUS][4] = (arrive, drained, released, 0) per CTA, where "drained" is after
# the CTA's s_waitcnt vmcnt(0) (its peer stores acknowledged). "wave": i64
# [barrier][CUS * 8 + wave][4] = (push, poll start, poll end, group) of the last
# all-reduce epilogue the wave ran in the stage that starts at that barrier.
ARPROBE = {"bar": None, "wave": None}


def drain_cta():
    """Wait for every memory op of every wave of the CTA (the s_waitcnt of grid_arrive,
    taken earlier so a stamp can follow it)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()


def st_stamps(addr, i, a, b):
    """Store the i64 pair (a, b) at i64 index i of addr."""
    a, b = fx.Int64(a), fx.Int64(b)
    st(
        fx.Vector.from_elements(
            [
                fx.Int32(a & 0xFFFFFFFF),
                fx.Int32(a >> 32),
                fx.Int32(b & 0xFFFFFFFF),
                fx.Int32(b >> 32),
            ],
            fx.Int32,
        ),
        rsrc(addr),
        i * 2,
    )
