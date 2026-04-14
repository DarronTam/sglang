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
from typing import TYPE_CHECKING

import torch

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


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
