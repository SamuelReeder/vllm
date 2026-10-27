# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.6-35B-A3B's gfx942 DecodeSpec against the conformance kit, plus its choices.

The kernel itself needs a gfx942 GPU and the 35B checkpoint; here it is a stub, so the
spec's decisions (which deployments, loaded models, caches and steps the kernel takes)
run anywhere.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.common.specialized_decode.kit import DecodeConformance, forward
from vllm.config.compilation import CUDAGraphMode
from vllm.models.common.specialized_decode.guards import select_decode_spec
from vllm.models.common.specialized_decode.spec import DecodeUnsupported
from vllm.models.qwen3_5_moe.amd.specialized_decode import spec as sd

SPECS = (sd.DECODE_SPEC,)
LAYERS = 40
ATTN = [i for i in range(LAYERS) if i % 4 == 3]
GDN = [i for i in range(LAYERS) if i % 4 != 3]
BLOCK = 528
SLOTS = 4
H = 2048


class _StubDecode:
    """The runtime's interface, without FlyDSL or the weights."""

    def __init__(self, vllm_config, lm, widths):
        layers = lm.model.layers
        self.attn_names = [layers[i].self_attn.attn.layer_name for i in ATTN]
        self.gdn_names = [layers[i].linear_attn.prefix for i in GDN]
        self.cache_layers = tuple(self.attn_names + self.gdn_names)
        self.widths = widths
        self.bound: dict[str, Any] | None = None
        self.attn_groups = self.attn_names[:1]
        self.gdn_groups = self.gdn_names[:3]

    def bind_caches(self, caches):
        self.bound = dict(caches)

    def run(self, step, input_ids, inputs_embeds, **kwargs):
        return ("kernel", step.kernel_width)


def _param(t: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(t, requires_grad=False)


def _fp8(n: int, k: int = H) -> SimpleNamespace:
    """A linear layer as vLLM's block FP8 method leaves it on gfx942."""
    w = torch.zeros(n, k, dtype=torch.uint8).view(torch.float8_e4m3fnuz)
    return SimpleNamespace(
        weight=_param(w), weight_scale_inv=_param(torch.ones(n // 128, k // 128))
    )


def _bf16(n: int, k: int = H) -> SimpleNamespace:
    return SimpleNamespace(weight=_param(torch.zeros(n, k, dtype=torch.bfloat16)))


def _experts(shuffled: bool = False) -> SimpleNamespace:
    """A MoE runner whose routed experts hold FP8 weights (stride-0 expert views: no
    memory)."""

    def w(*s):
        t = torch.zeros(1, *s, dtype=torch.uint8).view(torch.float8_e4m3fnuz)
        p = _param(t.expand(sd.NE, *s))
        p.is_shuffled = shuffled  # AITER's MoE shuffles its weights in place
        return p

    def s(*s):
        return _param(torch.ones(1, *s).expand(sd.NE, *s))

    routed = SimpleNamespace(
        w13_weight=w(2 * sd.FI, H),
        w13_weight_scale_inv=s(2 * sd.FI // 128, H // 128),
        w2_weight=w(H, sd.FI),
        w2_weight_scale_inv=s(H // 128, sd.FI // 128),
    )
    return SimpleNamespace(routed_experts=routed)


class _Layer(torch.nn.Module):
    def __init__(self, i: int, quantized: bool = True, **mlp):
        super().__init__()
        self.prefix = f"model.layers.{i}"
        lin = _fp8 if quantized else _bf16
        shared = mlp.pop(
            "shared_expert",
            SimpleNamespace(gate_up_proj=lin(2 * sd.FI), down_proj=lin(H, sd.FI)),
        )
        self.mlp = SimpleNamespace(
            gate=_bf16(sd.NE),
            shared_expert_gate=_bf16(1),
            shared_expert=shared,
            experts=mlp.pop("experts", _experts()),
        )
        mods = {
            "mlp.gate": self.mlp.gate,
            "mlp.shared_expert_gate": self.mlp.shared_expert_gate,
        }
        if shared is not None:
            mods |= {
                "mlp.shared_expert.gate_up_proj": shared.gate_up_proj,
                "mlp.shared_expert.down_proj": shared.down_proj,
            }
        if i % 4 == 3:
            self.self_attn = SimpleNamespace(
                attn=SimpleNamespace(layer_name=f"{self.prefix}.self_attn.attn")
            )
            mods |= {
                "self_attn.qkv_proj": lin(9216),
                "self_attn.o_proj": lin(H, 4096),
            }
        else:
            self.linear_attn = SimpleNamespace(prefix=f"{self.prefix}.linear_attn")
            mods |= {
                "linear_attn.in_proj_qkvz": lin(12288),
                "linear_attn.in_proj_ba": _bf16(2 * sd.NVH),
                "linear_attn.out_proj": lin(H, 4096),
            }
        self._mods = mods

    def get_submodule(self, target: str):
        return self._mods[target]


def _kv() -> torch.Tensor:
    """TRITON_ATTN's (blocks, heads, block, K | V) view; ROCM_ATTN's (blocks, K | V,
    block, heads HD) has the same shape with 2 KV heads."""
    return torch.zeros(3, 2, BLOCK, 2 * 256, dtype=torch.bfloat16)


def _state() -> tuple[torch.Tensor, torch.Tensor]:
    conv_bytes = 3 * sd.CONV_DIM * 2
    page = torch.zeros(SLOTS, conv_bytes + sd.NVH * 128 * 128 * 4, dtype=torch.uint8)
    conv = page[:, :conv_bytes].view(torch.bfloat16).view(SLOTS, 3, sd.CONV_DIM)
    ssm = page[:, conv_bytes:].view(torch.float32).view(SLOTS, sd.NVH, 128, 128)
    return conv, ssm


def _attn_layer(kv, backend: str = "ROCM_ATTN", interleaved_v: bool = False):
    return SimpleNamespace(
        kv_cache=kv,
        attn_backend=SimpleNamespace(get_name=lambda: backend),
        impl=SimpleNamespace(_use_interleaved_v_cache=interleaved_v),
    )


def make_config(**overrides) -> Any:
    text = SimpleNamespace(
        **sd.TEXT_CONFIG,
        rope_parameters={
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
        layer_types=[
            "full_attention" if i % 4 == 3 else "linear_attention"
            for i in range(LAYERS)
        ],
    )
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(specialized_decode="monokernel"),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
        ),
        model_config=SimpleNamespace(
            hf_text_config=text,
            dtype=torch.bfloat16,
            quantization="fp8",
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


def make_model(config, quantized: bool = True, **mlp) -> SimpleNamespace:
    layers = [_Layer(i, quantized, **dict(mlp)) for i in range(LAYERS)]
    sfc = config.compilation_config.static_forward_context
    for i in ATTN:
        sfc[layers[i].self_attn.attn.layer_name] = _attn_layer(_kv())
    for i in GDN:
        sfc[layers[i].linear_attn.prefix] = SimpleNamespace(kv_cache=_state())

    def named_parameters():
        for lay in layers:
            routed = lay.mlp.experts.routed_experts
            mods = dict(lay._mods, **{"mlp.experts.routed_experts": routed})
            for n, m in mods.items():
                for k, p in vars(m).items():
                    yield f"{lay.prefix}.{n}.{k}", p

    return SimpleNamespace(
        model=SimpleNamespace(layers=layers), named_parameters=named_parameters
    )


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


class TestQwen35MoeGfx942Spec(DecodeConformance):
    specs_module = "vllm.models.qwen3_5_moe.amd.specialized_decode.spec"

    @property
    def specs(self):
        return (sd.DECODE_SPEC,)

    def make_config(self, **overrides):
        return make_config(**overrides)

    def make_model(self, config):
        return make_model(config)

    def bad_caches(self, config, model):
        layers = model.model.layers
        conv, ssm = _state()
        return {
            layers[3].self_attn.attn.layer_name: _kv().to(torch.float16),
            layers[0].linear_attn.prefix: (conv.float(), ssm),
        }


def _decode(config=None, model=None, bind=True):
    config = config or make_config()
    return TestQwen35MoeGfx942Spec().make_decode(config, model, bind=bind)


def _md(decode, prefills=0, spec_decodes=0, query_len=1):
    gdn = SimpleNamespace(
        num_prefills=prefills,
        num_spec_decodes=spec_decodes,
        non_spec_state_indices_tensor=torch.zeros(8, dtype=torch.int32),
    )
    attn = SimpleNamespace(max_query_len=query_len)
    md = {n: attn for n in decode.kernel.attn_groups}
    return md | {n: gdn for n in decode.kernel.gdn_groups}


def test_decode_step_takes_the_kernel():
    decode = _decode()
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _md(decode)):
        step = decode.plan(torch.zeros(3, 8, dtype=torch.long))  # M-RoPE positions
    assert step is not None and step.kernel_width == 8


@pytest.mark.parametrize(
    "md", [dict(prefills=1), dict(spec_decodes=2), dict(query_len=2)]
)
def test_other_steps_take_vllm_path(md):
    decode = _decode()
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _md(decode, **md)):
        assert decode.plan(torch.zeros(4, dtype=torch.long)) is None


def test_width_outside_the_kernel_takes_vllm_path():
    decode = _decode()
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _md(decode)):
        assert decode.plan(torch.zeros(3, dtype=torch.long)) is None
        assert decode.plan(torch.zeros(16, dtype=torch.long)) is None


@pytest.mark.parametrize(
    "overrides, why",
    [
        (dict(parallel_config__tensor_parallel_size=2), "tensor parallel"),
        (dict(cache_config__cache_dtype="fp8"), "bf16 KV cache"),
        (dict(cache_config__mamba_ssm_cache_dtype="bfloat16"), "fp32 SSM"),
        (dict(model_config__dtype=torch.float16), "bf16 model dtype"),
        (dict(model_config__quantization=None), "FP8 checkpoint"),
        (dict(speculative_config=SimpleNamespace(method="mtp")), "speculative"),
    ],
)
def test_deployments_outside_the_kernel_refuse(overrides, why):
    with pytest.raises(DecodeUnsupported, match=why):
        select_decode_spec(make_config(**overrides), SPECS)


def test_dense_qwen38_config_refuses():
    from vllm.models.qwen3_5.amd.specialized_decode.spec import TEXT_CONFIG

    config = make_config()
    text = config.model_config.hf_text_config
    for key, value in TEXT_CONFIG.items():
        setattr(text, key, value)
    del text.num_experts
    with pytest.raises(DecodeUnsupported, match="num_experts"):
        select_decode_spec(config, SPECS)


def test_other_gpu_refuses(monkeypatch):
    monkeypatch.setattr(sd, "_gpu_arch", lambda: "gfx950")
    with pytest.raises(DecodeUnsupported, match="gfx942"):
        select_decode_spec(make_config(), SPECS)


@pytest.mark.parametrize(
    "model, why",
    [
        (dict(quantized=False), "in_proj_qkvz: needs FP8"),
        (dict(experts=_experts(shuffled=True)), "experts: needs FP8"),
        (dict(shared_expert=None), "fused shared expert"),
    ],
)
def test_models_outside_the_kernel_refuse(model, why):
    config = make_config()
    with pytest.raises(DecodeUnsupported, match=why):
        _decode(config, make_model(config, **model))


def test_kv_geometry_reads_triton_heads_first_view():
    """With 2 KV heads TRITON_ATTN's (blocks, heads, block, K | V) view has the shape
    of a (blocks, K | V, block, heads HD) one; the backend's order is the one read."""
    kv = _kv()
    block, s0, s_head, s_pos, s_v, xpack = sd.kv_geometry(
        kv, "TRITON_ATTN", sd.NKV, sd.HEAD_DIM
    )
    assert not xpack and block == BLOCK
    k = torch.arange(kv.numel(), dtype=torch.int64).view(kv.shape).flatten()
    want = torch.arange(kv.numel(), dtype=torch.int64).view(kv.shape)
    off = 2 * s0 + 1 * s_head + 77 * s_pos + 9  # block 2, head 1, position 77, dim 9
    assert k[off] == want[2, 1, 77, 9] and k[off + s_v] == want[2, 1, 77, 256 + 9]


def test_kv_geometry_reads_rocm_attn_paged_layout():
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    kv = _kv()
    k_cache, v_cache = PagedAttention.split_kv_cache(
        kv.transpose(0, 1), sd.NKV, sd.HEAD_DIM
    )
    block, s0, s_head, _, s_v, xpack = sd.kv_geometry(
        kv, "ROCM_ATTN", sd.NKV, sd.HEAD_DIM
    )
    assert xpack and block == BLOCK
    b, h, p, d = 2, 1, 300, 201
    k_off = b * s0 + h * s_head + (d // 8) * 8 * block + p * 8 + d % 8
    v_off = b * s0 + s_v + h * s_head + d * block + p
    assert k_cache[b, h, d // 8, p, d % 8].storage_offset() == k_off
    assert v_cache[b, h, d, p].storage_offset() == v_off


def test_interleaved_v_cache_refuses():
    config = make_config()
    decode = _decode(config, bind=False)
    sfc = config.compilation_config.static_forward_context
    name = decode.kernel.attn_names[0]
    sfc[name] = _attn_layer(sfc[name].kv_cache, interleaved_v=True)
    with pytest.raises(DecodeUnsupported, match="INTERLEAVED_V"):
        decode.bind_caches()


def test_qwen38_conv_state_refuses():
    """The dense 27B's GDN state (10240 conv channels, 48 V heads) is not this one."""
    config = make_config()
    decode = _decode(config, bind=False)
    sfc = config.compilation_config.static_forward_context
    name = decode.kernel.gdn_names[0]
    conv = torch.zeros(SLOTS, 3, 10240, dtype=torch.bfloat16)
    ssm = torch.zeros(SLOTS, 48, 128, 128)
    sfc[name] = SimpleNamespace(kv_cache=(conv, ssm))
    with pytest.raises(DecodeUnsupported, match="conv state .* SSM state"):
        decode.bind_caches()
