from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from rlvrs.schema import RolloutBatch, ScoredRolloutBatch


class BaseVerifier(ABC):
    """
    Base verifier interface.

    Input:
        RolloutBatch

    Output:
        ScoredRolloutBatch
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

    def score(self, rollout_batch: RolloutBatch) -> ScoredRolloutBatch:
        rewards = self.compute_rewards(rollout_batch)

        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            rewards = rewards.to(self.device, dtype=torch.float32)

        expected_batch_size = rollout_batch.batch_size
        if rewards.dim() == 1:
            if rewards.numel() != expected_batch_size:
                raise ValueError(
                    f"Verifier rewards shape mismatch: got {tuple(rewards.shape)}, "
                    f"expected [B] with B={expected_batch_size}."
                )
        elif rewards.dim() == 2:
            expected_num_prompts = rollout_batch.num_prompts
            expected_group_size = rollout_batch.group_size
            if rewards.shape != (expected_num_prompts, expected_group_size):
                raise ValueError(
                    f"Verifier rewards shape mismatch: got {tuple(rewards.shape)}, "
                    f"expected [{expected_num_prompts}, {expected_group_size}]."
                )
        else:
            raise ValueError(f"Rewards must be 1D or 2D, but got shape {tuple(rewards.shape)}.")

        return ScoredRolloutBatch(
            prompt_input_ids=rollout_batch.prompt_input_ids,
            prompt_attention_mask=rollout_batch.prompt_attention_mask,
            input_ids=rollout_batch.input_ids,
            attention_mask=rollout_batch.attention_mask,
            response_mask=rollout_batch.response_mask,
            old_logprobs=rollout_batch.old_logprobs,
            responses=rollout_batch.responses,
            prompts=rollout_batch.prompts,
            group_size=rollout_batch.group_size,
            num_prompts=rollout_batch.num_prompts,
            ref_logprobs=rollout_batch.ref_logprobs,
            extra=rollout_batch.extra,
            rewards=rewards,
        )

    @abstractmethod
    def compute_rewards(self, rollout_batch: RolloutBatch) -> torch.Tensor:
        """
        Return:
            rewards: [B] or [num_prompts, group_size]
        """
        raise NotImplementedError
