# Copyright (c) 2025 SandAI. All Rights Reserved.
#
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

from .dist_utils import dist_init, get_device, get_world_size, is_last_rank, is_last_tp_cp_rank
from .parallel_state import (
    destroy_model_parallel,
    get_cp_group,
    get_cp_rank,
    get_cp_world_size,
    get_dp_group,
    get_dp_group_gloo,
    get_dp_rank,
    get_dp_world_size,
    get_pipeline_model_parallel_first_rank,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pp_group,
    get_pp_rank,
    get_pp_world_size,
    get_tensor_model_parallel_last_rank,
    get_tensor_model_parallel_ranks,
    get_tensor_model_parallel_src_rank,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    is_initialized,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)

__all__ = [
    "dist_init",
    "is_initialized",
    "get_tp_group",
    "get_pp_group",
    "get_dp_group",
    "get_dp_group_gloo",
    "get_cp_group",
    "get_tp_world_size",
    "get_pp_world_size",
    "get_dp_world_size",
    "get_cp_world_size",
    "get_tp_rank",
    "get_pp_rank",
    "get_dp_rank",
    "get_cp_rank",
    "is_pipeline_first_stage",
    "is_pipeline_last_stage",
    "get_tensor_model_parallel_src_rank",
    "get_tensor_model_parallel_ranks",
    "get_tensor_model_parallel_last_rank",
    "get_pipeline_model_parallel_first_rank",
    "get_pipeline_model_parallel_last_rank",
    "get_pipeline_model_parallel_next_rank",
    "get_pipeline_model_parallel_prev_rank",
    "destroy_model_parallel",
    "is_last_rank",
    "is_last_tp_cp_rank",
    "get_world_size",
    "get_device",
]
