# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-27B's specialized decode on gfx942 (MI300X / MI300A / MI325X).

One persistent FlyDSL launch runs the whole decode step of the dense 27B text model
with its MXFP4 (OCP MX, e2m1 + e8m0) linear weights, for 1 to 8 decode rows on one
GPU. The kernels are in ``kernels/``; ``runtime/decode.py`` binds them to vLLM.
"""

from typing import TYPE_CHECKING

import torch

from vllm.forward_context import get_forward_context
from vllm.models.common.specialized_decode.spec import DecodeSpec

if TYPE_CHECKING:
    from .runtime.decode import Qwen35Gfx942Decode

MAX_ROWS = 8  # the kernel's norm prologues give each row one of a CTA's 8 waves
NKV, HEAD_DIM = 4, 256  # the attention layers' KV heads
# The dense Qwen3.8-27B text model the kernels are written for.
TEXT_CONFIG = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "vocab_size": 248320,
}


# The linear layers of a decoder layer the kernel streams, as vLLM fuses them.
GEMVS = {
    "gdn": {
        "inp": ("linear_attn.in_proj_qkvz", "linear_attn.in_proj_ba"),
        "out": ("linear_attn.out_proj",),
    },
    "attn": {"qkv": ("self_attn.qkv_proj",), "o": ("self_attn.o_proj",)},
    "mlp": {"gu": ("mlp.gate_up_proj",), "down": ("mlp.down_proj",)},
}


def mxfp4_parts(layer) -> tuple[torch.Tensor, torch.Tensor] | None:
    """A linear layer's packed e2m1 bytes [N, K/2] and e8m0 scales [N, K/32], or None
    when the layer does not hold MXFP4 weights."""
    w = getattr(layer, "weight", None)
    s = getattr(layer, "weight_scale", None)
    if w is None or s is None or w.dtype != torch.uint8 or s.dtype != torch.uint8:
        return None
    if w.dim() != 2 or s.shape != (w.shape[0], w.shape[1] // 16):
        return None
    return w, s


def attention_backend(layer) -> str:
    """The attention backend of an attention layer, as the KV cache layout depends on
    it; "ROCM_ATTN_INTERLEAVED_V" when AITER's fused KV write has interleaved its V."""
    name = layer.attn_backend.get_name()
    if getattr(getattr(layer, "impl", None), "_use_interleaved_v_cache", False):
        return name + "_INTERLEAVED_V"
    return name


def kv_geometry(
    kv: torch.Tensor, backend: str, nkv: int = NKV, head_dim: int = HEAD_DIM
) -> tuple[int, int, int, int, int, bool] | None:
    """(block, s0, s_head, s_pos, s_v, xpack) of an attention layer's 4-D KV view:
    the positions a block holds, the element strides of block, head and position, the
    elements from a K row to its V row, and whether the block interior is x-packed;
    None for a cache the kernel cannot address. nkv, head_dim: the layer's KV heads.

    TRITON_ATTN: the view is [blocks, NKV, block, 2 HD] (K | V per head) or [blocks, 2,
    block, NKV HD] (a K and a V half per block), with any strides; with 2 KV heads the
    two have the same shape, and the backend's (blocks, heads, block, 2 HD) order wins.
    ROCM_ATTN: the view is [blocks, 2, block, NKV HD], but each half holds the
    paged-attention layout: K [NKV][HD / 8][block][8] and V [NKV][HD][block]
    (ops/paged_attn.split_kv_cache)."""
    if kv.dim() != 4 or kv.stride(3) != 1:
        return None
    n = kv.shape[2]
    if backend == "ROCM_ATTN":
        if (
            kv.shape[1] != 2
            or kv.shape[3] != nkv * head_dim
            or kv.stride(2) != kv.shape[3]
        ):
            return None
        if n % 16 != 0:
            return None
        return n, kv.stride(0), head_dim * n, 8, kv.stride(1), True
    if backend != "TRITON_ATTN":
        return None
    if kv.shape[1] == nkv and kv.shape[3] == 2 * head_dim:
        return n, kv.stride(0), kv.stride(1), kv.stride(2), head_dim, False
    if kv.shape[1] == 2 and kv.shape[3] == nkv * head_dim:
        return n, kv.stride(0), head_dim, kv.stride(2), kv.stride(1), False
    return None


def _backends_missing() -> str | None:
    """Why FlyDSL or AITER's FlyDSL helpers cannot be imported, or None."""
    try:
        import flydsl  # noqa: F401
        from aiter.ops.flydsl.kernels import buffer_ops  # noqa: F401
    except ImportError as err:
        return str(err)
    return None


def _gpu_arch() -> str:
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    return props.gcnArchName.split(":")[0].lower()


def _widths(vllm_config) -> tuple[int, ...]:
    rows = min(MAX_ROWS, vllm_config.scheduler_config.max_num_seqs)
    return tuple(w for w in (1, 2, 4, 8) if w <= rows)


def _refuse_config(vllm_config) -> list[str]:
    """Every reason this deployment cannot run the kernel, before load."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    reasons = []
    text = vllm_config.model_config.hf_text_config
    for key, want in TEXT_CONFIG.items():
        if getattr(text, key, None) != want:
            reasons.append(
                f"needs Qwen3.8-27B ({key} {getattr(text, key, None)}, not {want})"
            )
    kinds = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]
    if list(getattr(text, "layer_types", [])) != kinds:
        reasons.append(
            "needs full attention in every 4th layer (3, 7, ...), "
            "linear attention elsewhere"
        )
    if vllm_config.parallel_config.tensor_parallel_size != 1:
        reasons.append("needs tensor parallel size 1")
    if vllm_config.model_config.dtype != torch.bfloat16:
        reasons.append(
            f"needs a bf16 model dtype, not {vllm_config.model_config.dtype}"
        )
    cache = vllm_config.cache_config
    if cache.cache_dtype not in ("auto", "bfloat16"):
        reasons.append(f"needs a bf16 KV cache, not {cache.cache_dtype}")
    if cache.mamba_ssm_cache_dtype not in ("auto", "float32"):
        reasons.append(f"needs an fp32 SSM state, not {cache.mamba_ssm_cache_dtype}")
    if cache.mamba_cache_dtype not in ("auto", "bfloat16"):
        reasons.append(f"needs a bf16 conv state, not {cache.mamba_cache_dtype}")
    if is_conv_state_dim_first():
        reasons.append(
            "needs the SD conv state layout (VLLM_SSM_CONV_STATE_LAYOUT unset)"
        )
    if _gpu_arch() != "gfx942":
        reasons.append(f"needs gfx942, not {_gpu_arch()}")
    missing = _backends_missing()
    if missing is not None:
        reasons.append(f"needs FlyDSL and AITER's FlyDSL helpers ({missing})")
    return reasons


def _refuse_model(lm) -> list[str]:
    """Every reason the loaded model cannot run the kernel: its linear layers must hold
    MXFP4 weights (for example amd/Qwen3.8-27B-Quark-AWQ-MXFP4)."""
    layer0 = lm.model.layers[0]
    names = GEMVS["gdn"]["inp"] + GEMVS["mlp"]["gu"]
    if any(mxfp4_parts(layer0.get_submodule(n)) is None for n in names):
        return ["needs MXFP4 linear weights (uint8 weight and weight_scale)"]
    return []


def _refuse_caches(vllm_config, decode: "Qwen35Gfx942Decode", caches) -> list[str]:
    """Every reason the kernel cannot address these caches."""
    reasons = []
    sfc = vllm_config.compilation_config.static_forward_context
    geometries = set()
    for name in decode.attn_names:
        kv = caches.get(name)
        if not isinstance(kv, torch.Tensor) or kv.dtype != torch.bfloat16:
            reasons.append(f"{name}: needs a bf16 KV tensor")
            continue
        backend = attention_backend(sfc[name])
        geometry = kv_geometry(kv, backend)
        if geometry is None:
            reasons.append(
                f"{name}: {backend} KV view {tuple(kv.shape)} strides {kv.stride()} "
                "(the kernel reads TRITON_ATTN's and ROCM_ATTN's layouts)"
            )
        geometries.add(geometry)
    if len(geometries) > 1:
        reasons.append("the attention layers' KV layouts differ")
    for name in decode.gdn_names:
        st = caches.get(name)
        if not isinstance(st, tuple) or len(st) != 2:
            reasons.append(f"{name}: needs a (conv, ssm) state pair")
            continue
        conv, ssm = st
        if (
            conv.dtype != torch.bfloat16
            or conv.shape[1:] != (3, 10240)
            or conv.stride()[1:] != (10240, 1)
        ):
            reasons.append(
                f"{name}: conv state {tuple(conv.shape)} {conv.dtype} "
                f"strides {conv.stride()}"
            )
        if (
            ssm.dtype != torch.float32
            or ssm.shape[1:] != (48, 128, 128)
            or not ssm[0].is_contiguous()
        ):
            reasons.append(
                f"{name}: SSM state {tuple(ssm.shape)} {ssm.dtype} "
                f"strides {ssm.stride()}"
            )
    return reasons


def _skip_step(vllm_config, decode: "Qwen35Gfx942Decode", width: int) -> str | None:
    """Why this step keeps vLLM's path: the kernel takes decode-only steps of one token
    a request."""
    md = get_forward_context().attn_metadata
    if not isinstance(md, dict):  # a profile / dummy run
        return "no attention metadata"
    attn = [md.get(n) for n in decode.attn_groups]
    gdn = [md.get(n) for n in decode.gdn_groups]
    if any(m is None for m in attn + gdn):
        return "layer metadata missing"
    if any(m.max_query_len != 1 for m in attn):
        return "not a decode-only step"
    if any(m.num_prefills != 0 or m.num_spec_decodes != 0 for m in gdn):
        return "not a decode-only step"
    if any(m.non_spec_state_indices_tensor is None for m in gdn):
        return "no state indices"
    return None


def _build(vllm_config, lm, widths: tuple[int, ...]) -> "Qwen35Gfx942Decode":
    from .runtime.decode import Qwen35Gfx942Decode

    return Qwen35Gfx942Decode(vllm_config, lm, widths)


DECODE_SPEC = DecodeSpec["Qwen35Gfx942Decode"](
    name="qwen38-27b-mxfp4-gfx942",
    supports=frozenset(),
    widths=_widths,
    refuse_config=_refuse_config,
    refuse_model=_refuse_model,
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=_build,
)
