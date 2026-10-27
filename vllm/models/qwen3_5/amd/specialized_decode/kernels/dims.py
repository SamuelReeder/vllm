# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-27B text-model dimensions (config.json text_config)."""

H = 5120  # hidden
FFN = 17408  # intermediate
VOCAB = 248320
NL = 64  # layers; layer i is full attention iff i % 4 == 3
EPS = 1e-6

# Gated DeltaNet (linear attention) layers
NKH, NVH, DK, DV = 16, 48, 128, 128
KEY_DIM, VAL_DIM = NKH * DK, NVH * DV  # 2048, 6144
CONV_DIM = 2 * KEY_DIM + VAL_DIM  # 10240 (q | k | v)
CONV_K = 4
# in_proj output columns: [q 2048 | k 2048 | v 6144 | z 6144 | b 48 | a 48]
INP_N = CONV_DIM + VAL_DIM + 2 * NVH  # 16480
Q_OFF, K_OFF, V_OFF, Z_OFF = 0, KEY_DIM, 2 * KEY_DIM, CONV_DIM
B_OFF, A_OFF = CONV_DIM + VAL_DIM, CONV_DIM + VAL_DIM + NVH

# Full-attention layers
NQ, NKV, HD = 24, 4, 256
ROT = 64
THETA = 1e7
# qkv output columns: [q|gate interleaved per head 12288 | k 1024 | v 1024]
QKV_N = 2 * NQ * HD + 2 * NKV * HD  # 14336

N_GDN = NL * 3 // 4  # 48
N_ATT = NL // 4  # 16


def is_attn(i: int) -> bool:
    return i % 4 == 3
