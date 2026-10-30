# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch backend of the gfx942 megakernel DecodeSpecs: FlyDSL JIT or hipDNN.

``VLLM_SPECIALIZED_DECODE_LAUNCH`` selects how a spec's kernel object launches a step:

- ``flydsl`` (default): the kernel object JIT-compiles its FlyDSL launcher and calls
  it. This path does not import hipdnn_vllm.
- ``hipdnn``: the kernel object builds no FlyDSL launcher. hipdnn_vllm's opaque custom
  op ``megakernel.<spec name>`` launches a prebuilt code object. A JSON payload of the
  binary key selects the code object; the kernel object's ``tab`` and ``glob`` tables
  are the inputs. Everything else (step checks, cache binding, weight packing, peer
  memory) stays in the kernel object.

The kernel object calls :meth:`HipdnnLaunch.prepare` when it binds the caches, which is
before CUDA graph capture. ``prepare`` builds one plan per width and agrees each result
over the TP group. A width without a plan on some rank keeps vLLM's path on every rank
(``skip_step``). No plan for any width raises ``DecodeUnsupported``: this module never
falls back to the FlyDSL JIT. A launch runs with ``strict``, so an execute failure
raises (megakernels ``docs/hipdnn-route.md`` H2, H3, H4, H7 and D10).
"""

import json
from collections.abc import Callable, Mapping

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.models.common.specialized_decode.guards import tp_agree, tp_group
from vllm.models.common.specialized_decode.spec import DecodeUnsupported

logger = init_logger(__name__)

FLYDSL, HIPDNN = "flydsl", "hipdnn"
LAUNCH_BACKENDS = (FLYDSL, HIPDNN)
PAYLOAD_SCHEMA = 1
# hipdnn_vllm's name of the op in HIPDNN_VLLM_OPS (opt-in, outside its default list)
OPT_IN = "custom_op"
_ENV = "VLLM_SPECIALIZED_DECODE_LAUNCH"


def launch_backend() -> str:
    """The launch backend of every megakernel DecodeSpec in this process."""
    return envs.VLLM_SPECIALIZED_DECODE_LAUNCH


def refuse_launch() -> list[str]:
    """Why the launch backend cannot run in this process, each with its fix; empty for
    ``flydsl``, which imports nothing here."""
    if launch_backend() != HIPDNN:
        return []
    try:
        from hipdnn_vllm.config import load_config
    except ImportError as err:
        return [
            f"{_ENV}=hipdnn needs the hipdnn_vllm package ({err}): install the "
            f"hipdnn_vllm gfx94X wheel, or unset {_ENV}"
        ]
    try:
        enabled = load_config().op_enabled(OPT_IN)
    except ValueError as err:  # hipdnn_vllm.config.ConfigError
        return [f"{_ENV}=hipdnn: hipdnn_vllm configuration: {err}"]
    if not enabled:
        return [
            f"{_ENV}=hipdnn needs hipdnn_vllm's {OPT_IN} op: add {OPT_IN} to "
            "HIPDNN_VLLM_OPS (and unset HIPDNN_VLLM_DISABLE)"
        ]
    return []


def kv_class(backend: str, kv: torch.Tensor, nkv: int, head_dim: int) -> str:
    """The KV layout class of an attention layer's cache: ``ROCM_ATTN`` (x-packed),
    ``TRITON_ATTN:1`` (view [blocks, nkv, block, 2 head_dim]) or ``TRITON_ATTN:2``
    (view [blocks, 2, block, nkv head_dim]). The TRITON forms are tested in the order
    of the specs' ``kv_geometry``, as the code-object exporter does
    (megakernels ``common/scripts/export_co.py`` ``kv_class``)."""
    if backend.startswith("ROCM_ATTN"):
        return backend
    if backend != "TRITON_ATTN":
        raise ValueError(f"no KV layout class for attention backend {backend}")
    if kv.shape[1] == nkv and kv.shape[3] == 2 * head_dim:
        return "TRITON_ATTN:1"
    if kv.shape[1] == 2 and kv.shape[3] == nkv * head_dim:
        return "TRITON_ATTN:2"
    raise ValueError(f"TRITON_ATTN view {tuple(kv.shape)} has no layout class")


def payload(
    model: str,
    cfg,
    compile_key: Callable[..., str],
    kernels_dir,
    *,
    arch: str,
    kv: str,
    block: int,
    tp: int,
    nlayers: int,
) -> str:
    """The custom op's payload for one width: the binary key of the code object
    (megakernels ``docs/hipdnn-route.md`` section 4) plus the launch's ``nlayers``.

    The fields are those of the exporter's manifest (megakernels
    ``common/scripts/export_co.py``, ``common/results/co/NOTES.md``): ``src`` is the
    model's ``compile_key(cfg.compiled(), <its kernels/ dir>)``, ``rt`` is
    ``Cfg.rt_geom`` (0 where unset), ``M``, ``cus`` and ``lm_head`` come from the
    ``Cfg``, ``tp`` is the TP size."""
    return json.dumps(
        {
            "schema": PAYLOAD_SCHEMA,
            "model": model,
            "src": compile_key(cfg.compiled(), kernels_dir),
            "arch": arch,
            "cus": int(cfg.cus),
            "kv": kv,
            "block": int(block),
            "M": int(cfg.M),
            "tp": int(tp),
            "rt": int(getattr(cfg, "rt_geom", 0) or 0),
            "lm_head": bool(cfg.lm_head),
            "nlayers": int(nlayers),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _ops() -> tuple[Callable, Callable]:
    """Load hipdnn_vllm's runtime in this worker (it registers
    ``torch.ops.hipdnn_vllm.*``) and return (custom_op_prepare, custom_op_exec)."""
    from hipdnn_vllm import _bootstrap

    _bootstrap.load()
    ops = torch.ops.hipdnn_vllm
    return ops.custom_op_prepare, ops.custom_op_exec


class HipdnnLaunch:
    """The launches of one spec's kernel object through hipdnn_vllm's custom op."""

    def __init__(self, model: str):
        """model: the DecodeSpec name; the op id is ``megakernel.<model>``. Raises
        ``DecodeUnsupported`` when hipdnn_vllm is missing or has not opted in."""
        reasons = refuse_launch()
        if reasons:
            raise DecodeUnsupported(
                f"Specialized decode {model}: " + "; ".join(reasons)
            )
        self.model = model
        self.op_id = f"megakernel.{model}"
        self._prepare, self._exec = _ops()
        self.payloads: dict[int, str] = {}
        self.declines: dict[int, str] = {}

    def prepare(self, payloads: Mapping[int, str], inputs: list[torch.Tensor]) -> None:
        """Build the plan of every width before CUDA graph capture.

        Collective over the TP group: every rank calls it with the same widths. A
        width that some rank declines is declined on every rank. Raises
        ``DecodeUnsupported`` when no width has a plan."""
        group = tp_group()[2]
        self.payloads = dict(payloads)
        self.declines = {}
        for w, p in sorted(self.payloads.items()):
            reason = self._prepare(self.op_id, p, inputs)
            if not tp_agree(not reason, group) and not reason:
                reason = "another TP rank declined"
            if reason:
                self.declines[w] = f"hipDNN has no plan for width {w}: {reason}"
                logger.warning(
                    "Specialized decode %s: width %d keeps vLLM's path: hipDNN "
                    "declined %s (%s); payload %s",
                    self.model,
                    w,
                    self.op_id,
                    reason,
                    p,
                )
            else:
                logger.info(
                    "Specialized decode %s: width %d launches through hipDNN %s; "
                    "payload %s",
                    self.model,
                    w,
                    self.op_id,
                    p,
                )
        if self.payloads and len(self.declines) == len(self.payloads):
            raise DecodeUnsupported(
                f"Specialized decode {self.model}: hipDNN has no plan for any width "
                f"({'; '.join(self.declines.values())}). Install the hipdnn_vllm "
                f"wheel whose code objects match this vLLM, or unset {_ENV}"
            )

    def declined(self, width: int) -> str | None:
        """Why this width keeps vLLM's path, or None when it has a plan."""
        return self.declines.get(width)

    def launch(self, width: int, inputs: list[torch.Tensor]) -> None:
        """Launch one step on the current stream; raises on any decline (H7)."""
        self._exec(self.op_id, self.payloads[width], inputs, True)
