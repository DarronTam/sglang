# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with Zeus graph capture and replay."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# Fields that must be int32 on Zeus device by the time replay starts. These
# are the entries that `populate_from_forward_batch` copies into static graph
# buffers; if any arrives as int64 the dtype-converting `copy_` falls into a
# CPU-bounce path that is *not* recorded into the captured graph, so the
# captured kernels read capture-time stale data on every replay (decode logits
# collapse and output sticks on a single repeated token). The right place to
# fix it is upstream — every constructor that targets a Zeus device must use
# `zeus_index_dtype(device)`. This guard catches regressions at the boundary.
_ZEUS_INT32_FORWARD_FIELDS = (
    "input_ids",
    "req_pool_indices",
    "positions",
    "mrope_positions",
    "seq_lens",
    "out_cache_loc",
)


def _check_zeus_int32_fields(forward_batch: ForwardBatch) -> None:
    offenders = []
    for name in _ZEUS_INT32_FORWARD_FIELDS:
        t = getattr(forward_batch, name, None)
        if t is None:
            continue
        if t.device.type != "zeus":
            continue
        if t.dtype != torch.int32:
            offenders.append(f"{name}: dtype={t.dtype}")
    if offenders:
        raise TypeError(
            "ZeusGraphRunner: forward_batch field(s) reached graph replay with "
            "non-int32 dtype on Zeus device — Zeus chip cannot operate on int64 "
            "device-side, and a dtype-converting `copy_` inside graph capture "
            "would silently mis-record. Route the construction through "
            "`zeus_index_dtype(device)` upstream. Offenders: " + ", ".join(offenders)
        )


class ZeusGraphRunner(CudaGraphRunner):
    """A ZeusGraphRunner runs the forward pass of a model with Zeus graph capture/replay."""

    def _create_device_graph(self):
        return self.device_module.ZEUSGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        # Zeus graph context manager uses 'zeus_graph=' kwarg (not 'cuda_graph=')
        with self.device_module.graph(zeus_graph=graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _cache_loc_dtype(self):
        # Zeus attention kernels use int32 indices
        return torch.int32

    def _index_dtype(self):
        # Zeus embedding and graph capture require int32 index buffers.
        return torch.int32

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        _check_zeus_int32_fields(forward_batch)
        return super().replay_prepare(forward_batch, pp_proxy_tensors)
