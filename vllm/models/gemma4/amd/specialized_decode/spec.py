# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 26B-A4B's specialized decode on gfx942 (MI300X / MI300A / MI325X).

One persistent FlyDSL launch runs the whole decode step of the 26B-A4B text model (25
sliding-window and 5 full attention layers, each followed by its dense GeGLU MLP and a
128-expert top-8 MoE in parallel) with its FP8 per-channel linear weights
(RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic), for 1 to 8 decode rows on one GPU. The
kernel returns the final-normed hidden states; vLLM's LM head applies the logit
soft-cap. The kernels are in ``kernels/``; ``runtime/decode.py`` binds them to vLLM.
The deployment checks shared with the Qwen3.5 family's kernels come from Qwen3.8's spec.
"""

from typing import TYPE_CHECKING

import torch

from vllm.models.common.specialized_decode.spec import DecodeSpec
from vllm.models.qwen3_5.amd.specialized_decode.spec import (
    _backends_missing,
    _gpu_arch,
    _skip_step,
    _widths,
    attention_backend,
    kv_geometry,
)

from .kernels.dims import (
    CONFIG,
    EFF,
    FFN,
    HD_F,
    HD_S,
    LAYER_TYPES,
    NE,
    NKV_F,
    NKV_S,
    NL,
    NQ,
    ROPE,
    VOCAB,
    H,
    is_full,
)

if TYPE_CHECKING:
    from .runtime.decode import Gemma4Gfx942Decode

# The text config's keys that differ by layer type: Transformers' Gemma 4 config keeps
# them in its per_layer_config (checked per layer below).
PER_LAYER = {
    "num_key_value_heads",
    "head_dim",
    "num_global_key_value_heads",
    "global_head_dim",
}
# Keys of the checkpoint config the kernel does not depend on in vLLM: vLLM runs the LM
# head itself (Cfg.lm_head False), and its text config of the Gemma 4 checkpoints reads
# tie_word_embeddings False although config.json says true (observed on
# RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic).
NOT_READ = {"tie_word_embeddings"}
# The FP8 linear layers the kernel repacks, as vLLM fuses them: (rows, columns) of
# each layer type. A full layer's qkv_proj holds [q | k | v = k] (attention_k_eq_v:
# vLLM loads k_proj into both shards); the kernel reads its q and k rows.
FP8_LINEARS = {
    "self_attn.qkv_proj": (
        (NQ * HD_S + 2 * NKV_S * HD_S, H),
        (NQ * HD_F + 2 * NKV_F * HD_F, H),
    ),
    "self_attn.o_proj": ((H, NQ * HD_S), (H, NQ * HD_F)),
    "mlp.gate_up_proj": ((2 * FFN, H), (2 * FFN, H)),
    "mlp.down_proj": ((H, FFN), (H, FFN)),
}


def fp8_parts(
    layer, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """A linear layer's e4m3fnuz weight as a [N, K] view and its fp32 per-row scales
    [N, 1], as vLLM keeps an FP8 per-channel layer on gfx942 (the checkpoint's e4m3fn
    bits with 0x80 -> 0 and the scales doubled; the weight a [K, N] transposed view, or
    [N, K] after AITER's per-token GEMM undid it), or None when the layer holds anything
    else (other quantization, or AITER's preshuffled layout). shape: (N, K)."""
    w = getattr(layer, "weight", None)
    s = getattr(layer, "weight_scale", None)
    if w is None or s is None or w.dtype != torch.float8_e4m3fnuz or w.dim() != 2:
        return None
    if getattr(layer, "aiter_bpreshuffled", False):
        return None
    if tuple(w.shape) == shape[::-1] and w.stride() == (1, shape[1]):
        w = w.t()
    if tuple(w.shape) != shape or w.stride() != (shape[1], 1):
        return None
    if s.dtype != torch.float32 or tuple(s.shape) != (shape[0], 1):
        return None
    return w, s


def expert_parts(
    experts,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """A MoE layer's routed experts' w13 [E, 2 EFF, H] ([gate; up]) and w2 [E, H, EFF]
    e4m3fnuz weights and their fp32 per-row scales [E, N, 1], unshuffled, or None when
    the layer holds anything else. experts: the decoder layer's ``experts``, the MoE
    runner that holds the routed experts' weights in its ``routed_experts``."""
    experts = getattr(experts, "routed_experts", None)
    parts = tuple(
        getattr(experts, n, None)
        for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    )
    if any(p is None for p in parts):
        return None
    w13, s13, w2, s2 = parts
    for w, s, shape in ((w13, s13, (2 * EFF, H)), (w2, s2, (H, EFF))):
        if w.dtype != torch.float8_e4m3fnuz or s.dtype != torch.float32:
            return None
        if getattr(w, "is_shuffled", False) or tuple(w.shape) != (NE, *shape):
            return None
        if w[0].stride() != (shape[1], 1):
            return None
        if tuple(s.shape) != (NE, shape[0], 1):
            return None
    return w13, s13, w2, s2


def _refuse_config(vllm_config) -> list[str]:
    """Every reason this deployment cannot run the kernel, before load."""
    reasons = []
    model_config = vllm_config.model_config
    text = model_config.hf_text_config
    for key, want in CONFIG.items():
        if key in PER_LAYER or key in NOT_READ:
            continue
        if getattr(text, key, None) != want:
            reasons.append(
                f"needs Gemma 4 26B-A4B ({key} {getattr(text, key, None)}, not {want})"
            )
    if list(getattr(text, "layer_types", None) or []) != LAYER_TYPES:
        reasons.append(
            "needs full attention in every 6th layer (5, 11, ...), "
            "sliding-window attention elsewhere"
        )
    else:
        # the attention dims differ by layer type: per layer in the config
        per_layer = getattr(text, "per_layer_config", None)
        for i in range(NL):
            want = (NKV_F, HD_F) if is_full(i) else (NKV_S, HD_S)
            lc = per_layer[i] if per_layer is not None else None
            got = tuple(
                getattr(lc, k, None) for k in ("num_key_value_heads", "head_dim")
            )
            if got != want:
                reasons.append(
                    f"needs Gemma 4 26B-A4B (layer {i} KV heads, head dim {got}, "
                    f"not {want})"
                )
                break
    rope = getattr(text, "rope_parameters", None) or {}
    for kind, params in ROPE.items():
        got = rope.get(kind) or {}
        for key, want in params.items():
            if got.get(key) != want:
                reasons.append(f"needs {kind} RoPE {key} {want}, not {got.get(key)}")
    if getattr(text, "attn_logit_softcapping", None) is not None:
        reasons.append("needs attention without logit soft-capping")
    quant = model_config.quantization
    if quant != "compressed-tensors":
        reasons.append(
            "needs the FP8 per-channel checkpoint "
            f"(RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic), not quantization {quant}"
        )
    if model_config.supports_multimodal_inputs or model_config.enable_prompt_embeds:
        # the runner then hands the step embeddings, already scaled, and no token ids:
        # the kernel embeds and scales the token ids itself
        reasons.append(
            "needs token ids (serve the text model: --language-model-only, "
            "no prompt embeddings)"
        )
    if vllm_config.parallel_config.tensor_parallel_size != 1:
        reasons.append("needs tensor parallel size 1")
    if model_config.dtype != torch.bfloat16:
        reasons.append(f"needs a bf16 model dtype, not {model_config.dtype}")
    cache = vllm_config.cache_config
    if cache.cache_dtype not in ("auto", "bfloat16"):
        reasons.append(f"needs a bf16 KV cache, not {cache.cache_dtype}")
    if _gpu_arch() != "gfx942":
        reasons.append(f"needs gfx942, not {_gpu_arch()}")
    missing = _backends_missing()
    if missing is not None:
        reasons.append(f"needs FlyDSL and AITER's FlyDSL helpers ({missing})")
    return reasons


def _refuse_model(lm) -> list[str]:
    """Every reason the loaded model cannot run the kernel: its linear layers and
    routed experts must hold FP8 per-channel weights in vLLM's plain (unshuffled) gfx942
    layout, and its router a bf16 matrix."""
    reasons = []
    for i, layer in enumerate(lm.model.layers):
        full = is_full(i)
        for name, shapes in FP8_LINEARS.items():
            shape = shapes[full]
            if fp8_parts(layer.get_submodule(name), shape) is None:
                reasons.append(
                    f"layer {i} {name}: needs FP8 (e4m3fnuz {shape[0]} x {shape[1]}, "
                    "fp32 per-row scales, unshuffled) weights"
                )
        if expert_parts(getattr(layer, "experts", None)) is None:
            reasons.append(
                f"layer {i} experts: needs FP8 (e4m3fnuz, fp32 per-row scales, "
                "unshuffled) w13 / w2 weights, as vLLM's Triton MoE keeps them"
            )
        w = getattr(layer.get_submodule("router.proj"), "weight", None)
        if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != (NE, H):
            reasons.append(f"layer {i} router.proj: needs a bf16 ({NE}, {H}) weight")
    # the kernel reads the embedding table and the final norm in place
    for name, shape in (("embed_tokens", (VOCAB, H)), ("norm", (H,))):
        w = getattr(getattr(lm.model, name, None), "weight", None)
        if (
            w is None
            or w.dtype != torch.bfloat16
            or tuple(w.shape) != shape
            or not w.is_contiguous()
        ):
            reasons.append(f"{name}: needs a contiguous bf16 {shape} weight")
    return reasons


def _refuse_caches(vllm_config, decode: "Gemma4Gfx942Decode", caches) -> list[str]:
    """Every reason the kernel cannot address these caches: one TRITON_ATTN geometry
    (any block size and strides, not x-packed) for each layer type."""
    reasons = []
    sfc = vllm_config.compilation_config.static_forward_context
    geometries: dict[bool, set] = {False: set(), True: set()}
    for i, name in enumerate(decode.attn_names):
        kv = caches.get(name)
        if not isinstance(kv, torch.Tensor) or kv.dtype != torch.bfloat16:
            reasons.append(f"{name}: needs a bf16 KV tensor")
            continue
        full = is_full(i)
        backend = attention_backend(sfc[name])
        nkv, hd = (NKV_F, HD_F) if full else (NKV_S, HD_S)
        geometry = kv_geometry(kv, backend, nkv, hd)
        if geometry is None or geometry[5]:
            reasons.append(
                f"{name}: {backend} KV view {tuple(kv.shape)} strides {kv.stride()} "
                "(the kernel reads TRITON_ATTN's layout, not an x-packed one)"
            )
            continue
        if geometry[0] % 16 != 0:
            reasons.append(f"{name}: KV block {geometry[0]} (needs a multiple of 16)")
        geometries[full].add(geometry)
    for full, kind in ((False, "sliding-window"), (True, "full")):
        if len(geometries[full]) > 1:
            reasons.append(f"the {kind} attention layers' KV layouts differ")
    return reasons


def _gemma_widths(vllm_config) -> tuple[int, ...]:
    """The widths the kernel serves: 1 and 2. On real text in vLLM (512-token prompts,
    MI300X) the kernel takes 3.57 / 4.20 / 5.20 / 6.94 ms per step at batch 1 / 2 / 4 /
    8 and vLLM's own path 4.41 / 4.64 / 5.22 / 6.12 ms: no gain at 4, a loss at 8
    (megakernels gemma4-26b-a4b-gfx942/docs/DESIGN.md, "Milestone 6"). Wider steps
    take vLLM's path with its CUDA graphs."""
    return tuple(w for w in _widths(vllm_config) if w <= 2)


def _build(vllm_config, lm, widths: tuple[int, ...]) -> "Gemma4Gfx942Decode":
    from .runtime.decode import Gemma4Gfx942Decode

    return Gemma4Gfx942Decode(vllm_config, lm, widths)


DECODE_SPEC = DecodeSpec["Gemma4Gfx942Decode"](
    name="gemma4-26b-a4b-fp8-gfx942",
    supports=frozenset(),
    widths=_gemma_widths,
    refuse_config=_refuse_config,
    refuse_model=_refuse_model,
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=_build,
)
