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

import logging

import torch


class GlobalLogger:
    _logger = None

    @classmethod
    def get_logger(cls, name=__name__, level=logging.INFO):
        if cls._logger is None:
            cls._logger = logging.getLogger("magi_logger")
            cls._logger.setLevel(logging.INFO)

            cls._logger.propagate = False
            cls._logger.handlers.clear()
            formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            cls._logger.addHandler(handler)

        return cls._logger


magi_logger = GlobalLogger.get_logger()


def print_per_rank(message):
    magi_logger.info(message)


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            magi_logger.info(message)
    else:
        magi_logger.info(message)
