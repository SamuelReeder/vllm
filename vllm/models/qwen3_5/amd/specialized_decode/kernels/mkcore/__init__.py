# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
# megakernels 46aa49d214ecda37714e9f9158c27c8baeaaf6ee, common/mkcore SRC_HASH 1e68a632f3838fdf
"""Shared FlyDSL gfx942 megakernel infrastructure of the megakernels models
(common/DESIGN.md). Imports no model module.

SRC_HASH: sha256 over the sorted relative paths and bytes of this package's .py files,
computed at import. key.compile_key puts it into every model's FlyDSL compile key, so an
edit of any mkcore file gives a new key (FlyDSL 0.4.0 does not follow helpers in other
directories)."""

import hashlib
import pathlib


def _src_hash() -> str:
    d = pathlib.Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(d.rglob("*.py")):
        h.update(str(p.relative_to(d)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


SRC_HASH = _src_hash()
