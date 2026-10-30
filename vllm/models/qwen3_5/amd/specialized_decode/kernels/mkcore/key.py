# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Compile key and kernel name of one megakernel build, and the Cfg fields that mkcore
reads (common/DESIGN.md D4, D5).

FlyDSL 0.4.0 keys its disk cache on the launcher's source and its scalar closure values
only: not on helpers called as module attributes (D.*), not on helpers in another
directory (mkcore), not on Cfg fields read as cfg.*, not on the kernel name. A model's
build binds `key = compile_key(...)` (a str) in the launcher closure and references it
in the launcher body, so FlyDSL's key holds it."""

import dataclasses
import hashlib
import pathlib
import zlib

from . import SRC_HASH

# Cfg fields that mkcore code reads (D4): every model's flat Cfg has them
COMMON_FIELDS = dict(M=int, cus=int, fence=bool, chaos=bool)


def check_cfg(cfg) -> None:
    """Assert that cfg has every COMMON_FIELDS field with its type."""
    names = {f.name for f in dataclasses.fields(cfg)}
    for name, t in COMMON_FIELDS.items():
        assert name in names, f"{type(cfg).__name__} lacks {name} (COMMON_FIELDS)"
        v = getattr(cfg, name)
        assert isinstance(v, t) and (t is bool or not isinstance(v, bool)), (name, v)


def source_hash(pkg_dir) -> str:
    """sha256 over the sorted relative paths and bytes of the .py files in pkg_dir."""
    d = pathlib.Path(pkg_dir).resolve()
    h = hashlib.sha256()
    for p in sorted(d.rglob("*.py")):
        h.update(str(p.relative_to(d)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def compile_key(cfg, pkg_dir, *extra) -> str:
    """Key of FlyDSL's disk cache for one build: repr((cfg.compiled(), *extra)) (extra:
    the model's debug globals that change the traced code), mkcore's SRC_HASH and the
    sources of the model's kernel package pkg_dir. cfg.compiled() is the Cfg the code
    object is built for (its run-time fields normalized, e.g. maxlen and maxblocks 0 at
    rt_geom 2), so Cfgs that differ in run-time fields only have one key; it is
    idempotent, so a caller may pass cfg or cfg.compiled()."""
    h = hashlib.sha256(repr((cfg.compiled(), *extra)).encode())
    h.update(SRC_HASH.encode())
    h.update(source_hash(pkg_dir).encode())
    return h.hexdigest()


def kernel_name(prefix: str, cfg, key: str) -> str:
    """`<prefix>_m<M>_<crc32 of the key>`: one name per build (not in FlyDSL's key)."""
    return f"{prefix}_m{cfg.M}_{zlib.crc32(key.encode()):08x}"
