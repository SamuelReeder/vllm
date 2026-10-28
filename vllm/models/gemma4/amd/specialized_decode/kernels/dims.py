# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""Gemma 4 26B-A4B text-model dimensions (config.json text_config). Checked against the
checkpoint's config at load time (reference.check_config)."""

import math

import torch

H = 2816  # hidden
NL = 30  # layers; layer i is full attention iff i % 6 == 5
VOCAB = 262144
EPS = 1e-6
SOFTCAP = 30.0  # final logits: SOFTCAP * tanh(logits / SOFTCAP)
# the embedding scale, cast to bf16 as vLLM and HF do: bf16(sqrt(2816)) = 53.0
EMBED_SCALE = float(torch.tensor(math.sqrt(H), dtype=torch.bfloat16))

NQ = 16  # q heads (both layer types)
# sliding-window layers
NKV_S, HD_S = 8, 256
WINDOW = 1024  # key j is visible to query i iff i - WINDOW < j <= i
THETA_S = 1e4  # NeoX RoPE over all HD_S dims
# full-attention layers: no v_proj (V = the k_proj output), "proportional" RoPE
NKV_F, HD_F = 2, 512
THETA_F = 1e6
ROT_F = 64  # rotated angle pairs: int(0.25 * 512 // 2); pairs (d, d + 256), d < 64

FFN = 2112  # dense MLP (GeGLU, gelu tanh)
NE, TOPK, EFF = 128, 8, 704  # routed experts, experts per token, expert size
# the kernel runs the dense MLP as NDS pseudo-experts of EFF columns (2112 = 3 x 704)
NDS = FFN // EFF
assert NDS * EFF == FFN
EFFP = 768  # EFF padded to the GEMV's 128-column superstep (down projection K)
QKV_S = NQ * HD_S + 2 * NKV_S * HD_S  # 8192: [q 4096 | k 2048 | v 2048]
QKV_F = NQ * HD_F + NKV_F * HD_F  # 9216: [q 8192 | k 1024]
O_S, O_F = NQ * HD_S, NQ * HD_F  # o_proj inputs: 4096, 8192


def is_full(i: int) -> bool:
    return i % 6 == 5


N_FULL = sum(is_full(i) for i in range(NL))  # 5
N_SLIDE = NL - N_FULL  # 25

CONFIG = dict(
    hidden_size=H,
    num_hidden_layers=NL,
    vocab_size=VOCAB,
    rms_norm_eps=EPS,
    final_logit_softcapping=SOFTCAP,
    num_attention_heads=NQ,
    num_key_value_heads=NKV_S,
    head_dim=HD_S,
    num_global_key_value_heads=NKV_F,
    global_head_dim=HD_F,
    sliding_window=WINDOW,
    intermediate_size=FFN,
    num_experts=NE,
    top_k_experts=TOPK,
    moe_intermediate_size=EFF,
    attention_k_eq_v=True,
    hidden_size_per_layer_input=0,
    num_kv_shared_layers=0,
    tie_word_embeddings=True,
    hidden_activation="gelu_pytorch_tanh",
    enable_moe_block=True,
)
LAYER_TYPES = [
    "full_attention" if is_full(i) else "sliding_attention" for i in range(NL)
]
ROPE = {
    "full_attention": dict(
        partial_rotary_factor=0.25, rope_theta=THETA_F, rope_type="proportional"
    ),
    "sliding_attention": dict(rope_theta=THETA_S, rope_type="default"),
}
