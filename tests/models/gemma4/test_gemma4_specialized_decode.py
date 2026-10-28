# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 26B-A4B's gfx942 DecodeSpec against the conformance kit, plus its choices.

The kernel itself needs a gfx942 GPU and the 26B checkpoint; here it is a stub and the
model's weights are meta tensors (the spec reads only their dtypes, shapes and
strides), so the spec's decisions (which deployments, loaded models, caches and steps
the kernel takes) run anywhere.
"""

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.common.specialized_decode.kit import DecodeConformance, forward
from vllm.config.compilation import CUDAGraphMode
from vllm.models.common.specialized_decode.guards import select_decode_spec
from vllm.models.common.specialized_decode.spec import DecodeUnsupported
from vllm.models.gemma4.amd.specialized_decode import spec as sd
from vllm.models.gemma4.amd.specialized_decode.kernels import dims

SPECS = (sd.DECODE_SPEC,)
H = dims.H
BLOCK_S, BLOCK_F = 16, 32  # the full group's block: twice the sliding one's


class _StubDecode:
    """The runtime's interface, without FlyDSL or the weights."""

    def __init__(self, vllm_config, lm, widths):
        layers = lm.model.layers
        self.attn_names = [layers[i].self_attn.attn.layer_name for i in range(dims.NL)]
        self.cache_layers = tuple(self.attn_names)
        self.widths = widths
        self.bound: dict[str, Any] | None = None
        # 5 sliding-window groups (layers g, g + 6, ...) and the full group
        self.attn_groups = self.attn_names[:6]
        self.gdn_groups: list[str] = []

    def bind_caches(self, caches):
        self.bound = dict(caches)

    def run(self, step, input_ids, inputs_embeds, **kwargs):
        return ("kernel", step.kernel_width)


def _param(t: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(t, requires_grad=False)


def _meta(*shape, dtype=torch.float8_e4m3fnuz) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, device="meta")


def _fp8(n: int, k: int, layout: str = "t") -> SimpleNamespace:
    """A linear layer as vLLM's FP8 per-channel method leaves it on gfx942: a [K, N]
    view of the [N, K] weight ("t", the Torch GEMM), [N, K] ("nk", AITER's per-token
    GEMM) or AITER's preshuffled one."""
    w = _meta(n, k)
    layer = SimpleNamespace(
        weight=_param(w.t() if layout == "t" else w),
        weight_scale=_param(_meta(n, 1, dtype=torch.float32)),
    )
    if layout == "preshuffled":
        layer.aiter_bpreshuffled = True
    return layer


def _bf16(*shape) -> SimpleNamespace:
    return SimpleNamespace(weight=_param(_meta(*shape, dtype=torch.bfloat16)))


def _experts(shuffled: bool = False) -> SimpleNamespace:
    """A MoE runner whose routed experts hold FP8 weights."""

    def w(*s):
        p = _param(_meta(dims.NE, *s))
        p.is_shuffled = shuffled  # AITER's MoE shuffles its weights in place
        return p

    def s(n):
        return _param(_meta(dims.NE, n, 1, dtype=torch.float32))

    routed = SimpleNamespace(
        w13_weight=w(2 * dims.EFF, H),
        w13_weight_scale=s(2 * dims.EFF),
        w2_weight=w(H, dims.EFF),
        w2_weight_scale=s(H),
    )
    return SimpleNamespace(routed_experts=routed)


class _Layer(torch.nn.Module):
    def __init__(self, i: int, layout: str = "t", **mods):
        super().__init__()
        self.prefix = f"language_model.model.layers.{i}"
        full = dims.is_full(i)
        self.self_attn = SimpleNamespace(
            attn=SimpleNamespace(layer_name=f"{self.prefix}.self_attn.attn")
        )
        self.experts = mods.pop("experts", _experts())
        linears = {
            name: _fp8(*shapes[full], layout) for name, shapes in sd.FP8_LINEARS.items()
        }
        self._mods = linears | {"router.proj": _bf16(dims.NE, H)} | mods

    def get_submodule(self, target: str):
        return self._mods[target]


def _kv(full: bool, block: int | None = None) -> torch.Tensor:
    """TRITON_ATTN's (blocks, heads, block, K | V) view of the default layout, which
    keeps a block's positions outermost (heads inside)."""
    nkv, hd = (dims.NKV_F, dims.HD_F) if full else (dims.NKV_S, dims.HD_S)
    block = block or (BLOCK_F if full else BLOCK_S)
    return torch.zeros(3, block, nkv, 2 * hd, dtype=torch.bfloat16).transpose(1, 2)


def _attn_layer(kv, backend: str = "TRITON_ATTN"):
    return SimpleNamespace(
        kv_cache=kv,
        attn_backend=SimpleNamespace(get_name=lambda: backend),
        impl=SimpleNamespace(),
    )


def make_config(**overrides) -> Any:
    text = SimpleNamespace(
        **{k: v for k, v in dims.CONFIG.items() if k not in sd.PER_LAYER},
        layer_types=list(dims.LAYER_TYPES),
        rope_parameters=copy.deepcopy(dims.ROPE),
        attn_logit_softcapping=None,
        per_layer_config=[
            SimpleNamespace(num_key_value_heads=dims.NKV_F, head_dim=dims.HD_F)
            if dims.is_full(i)
            else SimpleNamespace(num_key_value_heads=dims.NKV_S, head_dim=dims.HD_S)
            for i in range(dims.NL)
        ],
    )
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(specialized_decode="monokernel"),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(
            hf_text_config=text,
            dtype=torch.bfloat16,
            quantization="compressed-tensors",
            supports_multimodal_inputs=False,
            enable_prompt_embeds=False,
            max_model_len=4096,
            enable_sleep_mode=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            decode_context_parallel_size=1,
            use_ubatching=False,
            prefill_context_parallel_size=1,
        ),
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
            static_forward_context={},
        ),
        lora_config=None,
        kv_transfer_config=None,
        speculative_config=None,
        weight_transfer_config=None,
        use_v2_model_runner=True,
    )
    for path, value in overrides.items():
        obj = config
        *parents, leaf = path.split("__")
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, value)
    return config


def make_model(config, layouts: dict[int, str] | None = None, **mods):
    """layouts: the FP8 linear layers' layout of some layers (default "t")."""
    layouts = layouts or {}
    layers = [_Layer(i, layouts.get(i, "t"), **dict(mods)) for i in range(dims.NL)]
    sfc = config.compilation_config.static_forward_context
    for i, layer in enumerate(layers):
        sfc[layer.self_attn.attn.layer_name] = _attn_layer(_kv(dims.is_full(i)))
    backbone = SimpleNamespace(
        layers=layers,
        embed_tokens=_bf16(dims.VOCAB, H),
        norm=_bf16(H),
    )

    def named_parameters():
        for lay in layers:
            routed = lay.experts.routed_experts
            mods = dict(lay._mods, **{"experts.routed_experts": routed})
            for n, m in mods.items():
                for k, p in vars(m).items():
                    if isinstance(p, torch.nn.Parameter):
                        yield f"{lay.prefix}.{n}.{k}", p

    return SimpleNamespace(model=backbone, named_parameters=named_parameters)


@pytest.fixture(autouse=True)
def _gfx942(monkeypatch):
    monkeypatch.setattr(sd, "_gpu_arch", lambda: "gfx942")
    monkeypatch.setattr(sd, "_backends_missing", lambda: None)
    monkeypatch.setattr(
        sd,
        "DECODE_SPEC",
        sd.DECODE_SPEC.__class__(
            **{
                **vars(sd.DECODE_SPEC),
                "build": lambda cfg, lm, widths: _StubDecode(cfg, lm, widths),
            }
        ),
    )


class TestGemma4Gfx942Spec(DecodeConformance):
    specs_module = "vllm.models.gemma4.amd.specialized_decode.spec"

    @property
    def specs(self):
        return (sd.DECODE_SPEC,)

    def make_config(self, **overrides):
        return make_config(**overrides)

    def make_model(self, config):
        return make_model(config)

    def bad_caches(self, config, model):
        names = [lay.self_attn.attn.layer_name for lay in model.model.layers]
        return {
            names[0]: _kv(False).to(torch.float16),
            names[5]: torch.zeros(3, 4, BLOCK_F, 2 * dims.HD_F, dtype=torch.bfloat16),
        }


def _decode(config=None, model=None, bind=True):
    config = config or make_config()
    return TestGemma4Gfx942Spec().make_decode(config, model, bind=bind)


def _md(decode, query_len=1):
    attn = SimpleNamespace(max_query_len=query_len)
    return {n: attn for n in decode.kernel.attn_groups}


def test_decode_step_takes_the_kernel():
    decode = _decode()
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _md(decode)):
        step = decode.plan(torch.zeros(8, dtype=torch.long))
    assert step is not None and step.kernel_width == 8


def test_prefill_takes_vllm_path():
    decode = _decode()
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _md(decode, query_len=2)):
        assert decode.plan(torch.zeros(4, dtype=torch.long)) is None


def _set(path: str, value):
    """Overrides one hf_text_config entry (a dotted path into nested values)."""

    def apply(config):
        obj = config.model_config.hf_text_config
        *parents, leaf = path.split(".")
        for p in parents:
            obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
        if isinstance(obj, dict):
            obj[leaf] = value
        else:
            setattr(obj, leaf, value)
        return config

    return apply


def _layer_dims(config):
    config.model_config.hf_text_config.per_layer_config[5].head_dim = 256
    return config


@pytest.mark.parametrize(
    "overrides, edit, why",
    [
        (dict(parallel_config__tensor_parallel_size=2), None, "tensor parallel"),
        (dict(cache_config__cache_dtype="fp8"), None, "bf16 KV cache"),
        (dict(model_config__dtype=torch.float16), None, "bf16 model dtype"),
        (dict(model_config__quantization="fp8"), None, "FP8 per-channel"),
        (dict(model_config__supports_multimodal_inputs=True), None, "token ids"),
        (dict(model_config__enable_prompt_embeds=True), None, "token ids"),
        (dict(speculative_config=SimpleNamespace(method="mtp")), None, "speculative"),
        ({}, _set("sliding_window", 512), "sliding_window"),
        ({}, _set("num_experts", 64), "num_experts"),
        ({}, _set("attention_k_eq_v", False), "attention_k_eq_v"),
        ({}, _set("hidden_size_per_layer_input", 256), "hidden_size_per_layer"),
        ({}, _set("num_kv_shared_layers", 4), "num_kv_shared_layers"),
        ({}, _set("hidden_activation", "gelu"), "hidden_activation"),
        ({}, _set("layer_types", ["sliding_attention"] * 30), "every 6th layer"),
        ({}, _set("rope_parameters.full_attention.rope_theta", 1e4), "RoPE"),
        ({}, _set("attn_logit_softcapping", 50.0), "soft-capping"),
        ({}, _layer_dims, "layer 5 KV heads"),
    ],
)
def test_deployments_outside_the_kernel_refuse(overrides, edit, why):
    config = make_config(**overrides)
    if edit is not None:
        config = edit(config)
    with pytest.raises(DecodeUnsupported, match=why):
        select_decode_spec(config, SPECS)


def test_other_gpu_refuses(monkeypatch):
    monkeypatch.setattr(sd, "_gpu_arch", lambda: "gfx950")
    with pytest.raises(DecodeUnsupported, match="gfx942"):
        select_decode_spec(make_config(), SPECS)


@pytest.mark.parametrize(
    "model, why",
    [
        (dict(**{"self_attn.qkv_proj": _bf16(8192, H)}), "qkv_proj: needs FP8"),
        (dict(layouts={5: "preshuffled"}), "layer 5 self_attn.qkv_proj: needs FP8"),
        (dict(experts=_experts(shuffled=True)), "experts: needs FP8"),
        (dict(**{"router.proj": _fp8(dims.NE, H)}), "router.proj: needs a bf16"),
    ],
)
def test_models_outside_the_kernel_refuse(model, why):
    config = make_config()
    with pytest.raises(DecodeUnsupported, match=why):
        _decode(config, make_model(config, **model))


def test_fp8_layer_in_nk_layout_is_read():
    """AITER's per-token GEMM keeps the plain [N, K] weight; the Torch GEMM a [K, N]
    view of it. Both are the same bytes to the kernel."""
    config = make_config()
    assert _decode(config, make_model(config, layouts={0: "nk", 5: "nk"})) is not None
    w, s = sd.fp8_parts(_fp8(4096, H, "t"), (4096, H))
    assert w.shape == (4096, H) and w.stride() == (H, 1) and s.shape == (4096, 1)


def test_full_layer_qkv_must_hold_v_rows():
    """A full layer's qkv_proj in vLLM holds [q | k | v = k] (10240 rows)."""
    config = make_config()
    model = make_model(config)
    model.model.layers[5]._mods["self_attn.qkv_proj"] = _fp8(dims.QKV_F, H)
    with pytest.raises(DecodeUnsupported, match="layer 5 self_attn.qkv_proj"):
        _decode(config, model)


def _rebind(decode, config, i: int, kv, backend="TRITON_ATTN"):
    sfc = config.compilation_config.static_forward_context
    sfc[decode.kernel.attn_names[i]] = _attn_layer(kv, backend)


def test_sliding_layers_with_different_kv_layouts_refuse():
    config = make_config()
    decode = _decode(config, bind=False)
    _rebind(decode, config, 1, _kv(False, block=32))
    with pytest.raises(DecodeUnsupported, match="sliding-window attention .* differ"):
        decode.bind_caches()


def test_x_packed_kv_refuses():
    """ROCM_ATTN's paged layout: a [blocks, 2, block, NKV HD] view (the shape of the
    full layers' TRITON_ATTN view) whose halves are x-packed."""
    config = make_config()
    decode = _decode(config, bind=False)
    kv = torch.zeros(3, 2, BLOCK_F, dims.NKV_F * dims.HD_F, dtype=torch.bfloat16)
    _rebind(decode, config, 5, kv, "ROCM_ATTN")
    with pytest.raises(DecodeUnsupported, match="not an x-packed one"):
        decode.bind_caches()


def test_kv_geometry_of_both_layer_types():
    """The kernel's (block, s0, s_head, s_pos, s_v) of each layer type's view: K of
    (block 2, head h, position p, dim d) and its V, HD elements later."""
    for full, nkv, hd in (
        (False, dims.NKV_S, dims.HD_S),
        (True, dims.NKV_F, dims.HD_F),
    ):
        kv = _kv(full)
        block, s0, s_head, s_pos, s_v, xpack = sd.kv_geometry(
            kv, "TRITON_ATTN", nkv, hd
        )
        assert not xpack and block == kv.shape[2]
        want = torch.arange(kv.numel()).view(kv.shape[0], block, nkv, 2 * hd)
        want = want.transpose(1, 2)
        flat = torch.arange(kv.numel())
        h, p, d = nkv - 1, block - 3, hd - 5
        off = 2 * s0 + h * s_head + p * s_pos + d
        assert flat[off] == want[2, h, p, d]
        assert flat[off + s_v] == want[2, h, p, hd + d]
