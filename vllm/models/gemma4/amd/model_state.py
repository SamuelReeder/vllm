# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4's model runner state on ROCm: the default state plus the specialized
decode's cache binding."""

from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class Gemma4ROCMModelState(DefaultModelState):
    """Binds the specialized decode's caches for every cache the runner binds: the
    throwaway one memory profiling allocates, then the real one."""

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        super().initialize_kv_cache(kv_cache_config, block_tables)
        decode = getattr(self.model, "specialized_decode", None)
        if decode is not None:
            # the sliding-window and full attention layers have their own KV cache
            # groups: the kernel reads each layer's blocks from its own group
            decode.kernel.set_cache_groups(
                [g.layer_names for g in kv_cache_config.kv_cache_groups]
            )
            decode.bind_caches()
