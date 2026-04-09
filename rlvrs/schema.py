# schemas.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class TrainerType(str, Enum):
    GRPO = "grpo"
    RLOO = "rloo"
    GSPO = "gspo"


@dataclass
class RolloutBatch:
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor

    responses: List[str]
    prompts: List[str]

    group_size: int
    num_prompts: int

    ref_logprobs: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, Any]] = None

    @property
    def batch_size(self) -> int:
        return self.input_ids.size(0)

    @property
    def seq_len(self) -> int:
        return self.input_ids.size(1)


@dataclass
class ScoredRolloutBatch(RolloutBatch):
    rewards: Optional[torch.Tensor] = None


@dataclass
class TrainBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor

    ref_logprobs: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, Any]] = None
