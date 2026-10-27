# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-27B's gfx942 DecodeSpec against the conformance kit, plus its own choices.

The kernel itself needs a gfx942 GPU and the 27B checkpoint; here it is a stub, so the
spec's decisions (which deployments, caches and steps the kernel takes) run anywhere.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.models.common.specialized_decode.kit import DecodeConformance, forward
from vllm.config.compilation import CUDAGraphMode
from vllm.models.common.specialized_decode.guards import select_decode_spec
from vllm.models.common.specialized_decode.spec import DecodeUnsupported
from vllm.models.qwen3_5.amd.specialized_decode import spec as sd

SPECS = (sd.DECODE_SPEC,)
LAYERS = 64
ATTN = [i for i in range(LAYERS) if i % 4 == 3]
GDN = [i for i in range(LAYERS) if i % 4 != 3]
BLOCK = 784
SLOTS = 4


class _StubDecode:
    """The runtime's interface, without FlyDSL or the weights."""

    def __init__(self, vllm_config, lm, widths):
        layers = lm.model.layers
        self.attn_names = [layers[i].self_attn.attn.layer_name for i in ATTN]
        self.gdn_names = [layers[i].linear_attn.prefix for i in GDN]
        self.cache_layers = tuple(self.attn_names + self.gdn_names)
        self.widths = widths
        self.bound: dict[str, Any] | None = None
        # one attention cache group and three GDN groups, as vLLM groups Qwen3.8
        self.attn_groups = self.attn_names[:1]
        self.gdn_groups = self.gdn_names[:3]

    def bind_caches(self, caches):
        self.bound = dict(caches)

    def run(self, step, input_ids, inputs_embeds, **kwargs):
        return ("kernel", step.kernel_width)


def _mxfp4(n: int, k: int) -> SimpleNamespace:
    return SimpleNamespace(
        weight=torch.nn.Parameter(
            torch.zeros(n, k // 2, dtype=torch.uint8), requires_grad=False
        ),
        weight_scale=torch.nn.Parameter(
            torch.zeros(n, k // 32, dtype=torch.uint8), requires_grad=False
        ),
    )


class _Layer(torch.nn.Module):
    def __init__(self, i: int, quantized: bool = True):
        super().__init__()
        self.prefix = f"model.layers.{i}"
        lin = _mxfp4 if quantized else (lambda n, k: torch.nn.Linear(k, n, bias=False))
        mods = {"mlp.gate_up_proj": lin(32, 128)}
        if i % 4 == 3:
            self.self_attn = SimpleNamespace(
                attn=SimpleNamespace(layer_name=f"{self.prefix}.self_attn.attn")
            )
        else:
            self.linear_attn = SimpleNamespace(prefix=f"{self.prefix}.linear_attn")
            mods |= {
                "linear_attn.in_proj_qkvz": lin(32, 128),
                "linear_attn.in_proj_ba": lin(16, 128),
            }
        self._mods = mods

    def get_submodule(self, target: str):
        return self._mods[target]


def _kv(layout: str = "planar") -> torch.Tensor:
    if layout == "planar":  # [blocks, K|V, block, NKV HD]
        return torch.zeros(3, 2, BLOCK, 4 * 256, dtype=torch.bfloat16)
    return torch.zeros(
        3, 4, BLOCK, 512, dtype=torch.bfloat16
    )  # [blocks, NKV, block, K|V]


def _state() -> tuple[torch.Tensor, torch.Tensor]:
    page = torch.zeros(SLOTS, 3 * 10240 * 2 + 48 * 128 * 128 * 4, dtype=torch.uint8)
    conv = page[:, : 3 * 10240 * 2].view(torch.bfloat16).view(SLOTS, 3, 10240)
    ssm = page[:, 3 * 10240 * 2 :].view(torch.float32).view(SLOTS, 48, 128, 128)
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


def make_model(config, quantized: bool = True) -> SimpleNamespace:
    layers = [_Layer(i, quantized) for i in range(LAYERS)]
    sfc = config.compilation_config.static_forward_context
    for i in ATTN:
        sfc[layers[i].self_attn.attn.layer_name] = _attn_layer(_kv())
    for i in GDN:
        sfc[layers[i].linear_attn.prefix] = SimpleNamespace(kv_cache=_state())
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers),
        named_parameters=lambda: [
            (f"{n}.{k}", p)
            for lay in layers
            for n, m in lay._mods.items()
            for k, p in vars(m).items()
        ],
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


class TestQwen35Gfx942Spec(DecodeConformance):
    specs_module = "vllm.models.qwen3_5.amd.specialized_decode.spec"

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


def _decode(config=None):
    config = config or make_config()
    return TestQwen35Gfx942Spec().make_decode(config)


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
        step = decode.plan(torch.zeros(4, dtype=torch.long))
    assert step is not None and step.kernel_width == 4


@pytest.mark.parametrize(
    "md", [dict(prefills=1), dict(spec_decodes=2), dict(query_len=2)]
)
def test_other_steps_take_vllm_path(md):
    decode = _decode()
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _md(decode, **md)):
        assert decode.plan(torch.zeros(4, dtype=torch.long)) is None


def test_profile_run_takes_vllm_path():
    decode = _decode()
    with forward(decode.vllm_config, CUDAGraphMode.FULL, None):
        assert decode.plan(torch.zeros(1, dtype=torch.long)) is None


def test_width_outside_the_kernel_takes_vllm_path():
    decode = _decode()
    with forward(decode.vllm_config, CUDAGraphMode.FULL, _md(decode)):
        assert decode.plan(torch.zeros(3, dtype=torch.long)) is None
        assert decode.plan(torch.zeros(16, dtype=torch.long)) is None


@pytest.mark.parametrize(
    "seqs, widths", [(8, (1, 2, 4, 8)), (4, (1, 2, 4)), (256, (1, 2, 4, 8))]
)
def test_widths_follow_max_num_seqs(seqs, widths):
    assert sd._widths(make_config(scheduler_config__max_num_seqs=seqs)) == widths


@pytest.mark.parametrize(
    "overrides",
    [
        dict(parallel_config__tensor_parallel_size=2),
        dict(cache_config__cache_dtype="fp8"),
        dict(cache_config__mamba_ssm_cache_dtype="bfloat16"),
        dict(model_config__dtype=torch.float16),
    ],
)
def test_deployments_outside_the_kernel_refuse(overrides):
    with pytest.raises(DecodeUnsupported):
        select_decode_spec(make_config(**overrides), SPECS)


def test_other_model_refuses():
    config = make_config()
    config.model_config.hf_text_config.num_hidden_layers = 40
    with pytest.raises(DecodeUnsupported, match="num_hidden_layers"):
        select_decode_spec(config, SPECS)


def test_other_gpu_refuses(monkeypatch):
    monkeypatch.setattr(sd, "_gpu_arch", lambda: "gfx950")
    with pytest.raises(DecodeUnsupported, match="gfx942"):
        select_decode_spec(make_config(), SPECS)


def test_ds_conv_layout_refuses(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first",
        lambda: True,
    )
    with pytest.raises(DecodeUnsupported, match="conv state layout"):
        select_decode_spec(make_config(), SPECS)


def test_unquantized_model_refuses():
    config = make_config()
    model = make_model(config, quantized=False)
    with pytest.raises(DecodeUnsupported, match="MXFP4"):
        TestQwen35Gfx942Spec().make_decode(config, model)


@pytest.mark.parametrize("layout", ["planar", "per_head"])
def test_kv_geometry_reads_triton_layouts(layout):
    kv = _kv(layout)
    block, s0, s_head, s_pos, s_v, xpack = sd.kv_geometry(kv, "TRITON_ATTN")
    assert not xpack
    k = torch.arange(kv.numel(), dtype=torch.int64).view(kv.shape)
    flat = k.flatten()
    # block 1, head 2, position 5, dim 7: its K and its V
    if layout == "planar":
        want_k, want_v = k[1, 0, 5, 2 * 256 + 7], k[1, 1, 5, 2 * 256 + 7]
    else:
        want_k, want_v = k[1, 2, 5, 7], k[1, 2, 5, 256 + 7]
    off = 1 * s0 + 2 * s_head + 5 * s_pos + 7
    assert block == BLOCK
    assert flat[off] == want_k and flat[off + s_v] == want_v


def test_kv_geometry_reads_rocm_attn_paged_layout():
    """ROCM_ATTN's halves hold K [NKV][HD / 8][block][8] and V [NKV][HD][block], as
    vllm/v1/attention/ops/paged_attn.split_kv_cache views them."""
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    kv = _kv()
    k_cache, v_cache = PagedAttention.split_kv_cache(kv.transpose(0, 1), 4, 256)
    block, s0, s_head, _, s_v, xpack = sd.kv_geometry(kv, "ROCM_ATTN")
    assert xpack and block == BLOCK
    b, h, p, d = 2, 3, 100, 41
    k_off = b * s0 + h * s_head + (d // 8) * 8 * block + p * 8 + d % 8
    v_off = b * s0 + s_v + h * s_head + d * block + p
    assert k_cache[b, h, d // 8, p, d % 8].storage_offset() == k_off
    assert v_cache[b, h, d, p].storage_offset() == v_off


def test_kv_geometry_refuses_other_views():
    assert (
        sd.kv_geometry(
            torch.zeros(3, 8, BLOCK, 256, dtype=torch.bfloat16), "TRITON_ATTN"
        )
        is None
    )
    assert sd.kv_geometry(_kv().transpose(2, 3), "ROCM_ATTN") is None
    assert sd.kv_geometry(_kv(), "ROCM_AITER_FA") is None


def test_interleaved_v_cache_refuses():
    config = make_config()
    model = make_model(config)
    decode = TestQwen35Gfx942Spec().make_decode(config, model, bind=False)
    sfc = config.compilation_config.static_forward_context
    name = decode.kernel.attn_names[0]
    sfc[name] = _attn_layer(sfc[name].kv_cache, interleaved_v=True)
    with pytest.raises(DecodeUnsupported, match="INTERLEAVED_V"):
        decode.bind_caches()
