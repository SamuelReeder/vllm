# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.6-35B-A3B's specialized decode on gfx942 (MI300X / MI300A / MI325X).

One persistent FlyDSL launch runs the whole decode step of the 35B-A3B text model (30
Gated DeltaNet and 10 gated attention layers, each followed by a 256-expert top-8 MoE
with a shared expert) with its FP8 block-scaled linear weights, for 1 to 8 decode rows
on one GPU. The kernels are in ``kernels/``; ``runtime/decode.py`` binds them to vLLM.
The deployment, step and cache checks the model shares with Qwen3.8-27B's kernel come
from its spec.
"""

from typing import TYPE_CHECKING

import torch

from vllm.models.common.specialized_decode.hipdnn_launch import refuse_launch
from vllm.models.common.specialized_decode.spec import DecodeSpec
from vllm.models.qwen3_5.amd.specialized_decode.spec import (
    _backends_missing,
    _gpu_arch,
    _widths,
    attention_backend,
    kv_geometry,
)
from vllm.models.qwen3_5.amd.specialized_decode.spec import (
    _skip_step as _decode_only_skip_step,
)

if TYPE_CHECKING:
    from .runtime.decode import Qwen35MoeGfx942Decode

NAME = "qwen36-35b-a3b-fp8-gfx942"  # also the hipDNN op id's model (hipdnn_launch.py)
LAYERS = 40  # layer i is full attention iff i % 4 == 3
NKV, HEAD_DIM = 2, 256  # the attention layers' KV heads
CONV_DIM, NVH, DK = 8192, 32, 128  # GDN conv channels (q | k | v), V heads, head dim
NE, FI = 256, 512  # routed experts, expert intermediate size
# The Qwen3.6-35B-A3B text model the kernels are written for.
TEXT_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": 16,
    "num_key_value_heads": NKV,
    "head_dim": HEAD_DIM,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": NVH,
    "linear_key_head_dim": DK,
    "linear_value_head_dim": DK,
    "linear_conv_kernel_dim": 4,
    "vocab_size": 248320,
    "rms_norm_eps": 1e-6,
    "num_experts": NE,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": FI,
    "shared_expert_intermediate_size": 512,
    "attn_output_gate": True,
    "tie_word_embeddings": False,
}
ROPE = {"rope_theta": 1e7, "partial_rotary_factor": 0.25}

# The FP8 block-scaled linear layers the kernel repacks, as vLLM fuses them.
FP8_LINEARS = {
    "gdn": ("linear_attn.in_proj_qkvz", "linear_attn.out_proj"),
    "attn": ("self_attn.qkv_proj", "self_attn.o_proj"),
    "mlp": ("mlp.shared_expert.gate_up_proj", "mlp.shared_expert.down_proj"),
}


def fp8_parts(layer) -> tuple[torch.Tensor, torch.Tensor] | None:
    """A linear layer's e4m3fnuz weight [N, K] and fp32 128 x 128 block scales [N/128,
    K/128] as vLLM keeps them on gfx942 (the checkpoint's e4m3fn bits with 0x80 -> 0
    and the scales doubled), or None when the layer holds anything else."""
    w = getattr(layer, "weight", None)
    s = getattr(layer, "weight_scale_inv", None)
    if w is None or s is None or w.dtype != torch.float8_e4m3fnuz or w.dim() != 2:
        return None
    if getattr(layer, "aiter_bpreshuffled", False):
        return None
    n, k = w.shape
    if s.dtype != torch.float32 or s.shape != (-(-n // 128), -(-k // 128)):
        return None
    return w, s


def expert_parts(
    experts,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """A MoE layer's routed experts' w13 [E, 2 FI, H] ([gate; up]) and w2 [E, H, FI]
    e4m3fnuz weights and their fp32 block scales, unshuffled, or None when the layer
    holds anything else. experts: ``mlp.experts``, the MoE runner that holds the
    routed experts' weights in its ``routed_experts``."""
    experts = getattr(experts, "routed_experts", None)
    parts = tuple(
        getattr(experts, n, None)
        for n in (
            "w13_weight",
            "w13_weight_scale_inv",
            "w2_weight",
            "w2_weight_scale_inv",
        )
    )
    if any(p is None for p in parts):
        return None
    w13, s13, w2, s2 = parts
    for w, s in ((w13, s13), (w2, s2)):
        if w.dtype != torch.float8_e4m3fnuz or s.dtype != torch.float32:
            return None
        if getattr(w, "is_shuffled", False) or w.dim() != 3 or w.shape[0] != NE:
            return None
        if s.shape != (NE, -(-w.shape[1] // 128), -(-w.shape[2] // 128)):
            return None
    if w13.shape[1:] != (2 * FI, w2.shape[1]) or w2.shape[2] != FI:
        return None
    return w13, s13, w2, s2


def _refuse_config(vllm_config) -> list[str]:
    """Every reason this deployment cannot run the kernel, before load."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    reasons = []
    text = vllm_config.model_config.hf_text_config
    for key, want in TEXT_CONFIG.items():
        if getattr(text, key, None) != want:
            reasons.append(
                f"needs Qwen3.6-35B-A3B ({key} {getattr(text, key, None)}, not {want})"
            )
    rope = getattr(text, "rope_parameters", None) or {}
    for key, want in ROPE.items():
        if rope.get(key) != want:
            reasons.append(f"needs RoPE {key} {want}, not {rope.get(key)}")
    if not getattr(text, "norm_topk_prob", True):
        reasons.append("needs renormalized top-k routing weights (norm_topk_prob)")
    kinds = [
        "full_attention" if i % 4 == 3 else "linear_attention" for i in range(LAYERS)
    ]
    if list(getattr(text, "layer_types", [])) != kinds:
        reasons.append(
            "needs full attention in every 4th layer (3, 7, ...), "
            "linear attention elsewhere"
        )
    quant = vllm_config.model_config.quantization
    if quant != "fp8":
        reasons.append(
            f"needs the FP8 checkpoint (Qwen/Qwen3.6-35B-A3B-FP8), not quantization "
            f"{quant}"
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
    return reasons + refuse_launch()


def _refuse_model(lm) -> list[str]:
    """Every reason the loaded model cannot run the kernel: its linear layers must hold
    FP8 128 x 128 block-scaled weights in vLLM's plain (unshuffled) gfx942 layout, its
    shared expert must be its own MLP, and the router and the shared expert's gate bf16
    matrices."""
    reasons = []
    for i, layer in enumerate(lm.model.layers):
        kind = "attn" if i % 4 == 3 else "gdn"
        mlp = layer.mlp
        if getattr(mlp, "shared_expert", None) is None:
            reasons.append(
                f"layer {i}: needs the shared expert as its own MLP "
                "(AITER's fused shared expert is on)"
            )
            continue
        for name in FP8_LINEARS[kind] + FP8_LINEARS["mlp"]:
            if fp8_parts(layer.get_submodule(name)) is None:
                reasons.append(
                    f"layer {i} {name}: needs FP8 (e4m3fnuz, 128 x 128 fp32 block "
                    "scales, unshuffled) weights"
                )
        if expert_parts(mlp.experts) is None:
            reasons.append(
                f"layer {i} experts: needs FP8 (e4m3fnuz, 128 x 128 fp32 block "
                "scales, unshuffled) w13 / w2 weights, as vLLM's Triton MoE keeps them"
            )
        bf16 = [("mlp.gate", (NE, 2048)), ("mlp.shared_expert_gate", (1, 2048))]
        if kind == "gdn":
            bf16.append(("linear_attn.in_proj_ba", (2 * NVH, 2048)))
        for name, shape in bf16:
            w = getattr(layer.get_submodule(name), "weight", None)
            if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != shape:
                reasons.append(f"layer {i} {name}: needs a bf16 {shape} weight")
    return reasons


def _refuse_caches(vllm_config, decode: "Qwen35MoeGfx942Decode", caches) -> list[str]:
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
        geometry = kv_geometry(kv, backend, NKV, HEAD_DIM)
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
            or conv.shape[1:] != (3, CONV_DIM)
            or conv.stride()[1:] != (CONV_DIM, 1)
        ):
            reasons.append(
                f"{name}: conv state {tuple(conv.shape)} {conv.dtype} "
                f"strides {conv.stride()}"
            )
        if (
            ssm.dtype != torch.float32
            or ssm.shape[1:] != (NVH, DK, DK)
            or not ssm[0].is_contiguous()
        ):
            reasons.append(
                f"{name}: SSM state {tuple(ssm.shape)} {ssm.dtype} "
                f"strides {ssm.stride()}"
            )
    return reasons


def _skip_step(vllm_config, decode: "Qwen35MoeGfx942Decode", width: int) -> str | None:
    """Why this step keeps vLLM's path: a width without a launch (hipDNN can decline a
    width at warm-up), or a step that is not decode-only (Qwen3.8's checks)."""
    if decode.hipdnn is not None and (why := decode.hipdnn.declined(width)):
        return why
    return _decode_only_skip_step(vllm_config, decode, width)


def _build(vllm_config, lm, widths: tuple[int, ...]) -> "Qwen35MoeGfx942Decode":
    from .runtime.decode import Qwen35MoeGfx942Decode

    return Qwen35MoeGfx942Decode(vllm_config, lm, widths)


DECODE_SPEC = DecodeSpec["Qwen35MoeGfx942Decode"](
    name=NAME,
    supports=frozenset(),
    widths=_widths,
    refuse_config=_refuse_config,
    refuse_model=_refuse_model,
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=_build,
)
