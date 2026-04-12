from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim

from rlvrs.schema import ScoredRolloutBatch

from .base import BaseTrainer
from .utils import get_response_logprobs, masked_mean


class GRPOTrainer(BaseTrainer):
    def __init__(
        self,
        actor: nn.Module,
        verifier,
        optimizer: optim.Optimizer,
        config: None,
        rollout_engine=None,
        lr_scheduler=None,
        device=None,
    ):
        super().__init__(
            actor=actor,
            verifier=verifier,
            optimizer=optimizer,
            config=config,
            rollout_engine=rollout_engine,
            lr_scheduler=lr_scheduler,
            device=device,
        )

        self.clip_range = self.config.get("clip_range", 0.2)
        self.kl_coef = self.config.get("kl_coef", 0.0)
        self.eps = self.config.get("eps", 1e-8)
        self.reward_scale = self.config.get("reward_scale", 1.0)
        self.normalize_advantages = self.config.get("normalize_advantages", True)

    def build_train_batch(self, scored_batch: ScoredRolloutBatch):
        # train_batch = dict(scored_batch)

        # rewards = train_batch["rewards"]
        rewards = scored_batch.rewards
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            rewards = rewards.to(self.device)

        rewards = rewards * self.reward_scale

        # Handle grouped rewards:
        # case 1: rewards is [num_prompts, group_size]
        # case 2: rewards is [B] and group_size is provided
        # case 3: rewards is already flattened and no normalization grouping available
        if rewards.dim() == 2:
            group_rewards = rewards
            flat_rewards = rewards.reshape(-1)
            advantages = self._compute_group_advantages(group_rewards).reshape(-1)
        elif rewards.dim() == 1:
            group_size = (
                scored_batch.group_size
                if hasattr(scored_batch, "group_size")
                else self.config.get("group_size", None)
            )
            if group_size is not None:
                if rewards.numel() % group_size != 0:
                    raise ValueError(
                        f"rewards.numel()={rewards.numel()} is not divisible by group_size={group_size}"
                    )
                group_rewards = rewards.view(-1, group_size)
                flat_rewards = rewards
                advantages = self._compute_group_advantages(group_rewards).reshape(-1)
            else:
                flat_rewards = rewards
                if self.normalize_advantages:
                    mean = rewards.mean()
                    std = rewards.std(unbiased=False).clamp_min(self.eps)
                    advantages = (rewards - mean) / std
                else:
                    advantages = rewards
        else:
            raise ValueError("rewards must be 1D or 2D tensor")

        train_batch = asdict(scored_batch)
        train_batch["rewards"] = flat_rewards
        train_batch["advantages"] = advantages.to(self.device)

        # move common tensors to device
        for key in [
            "input_ids",
            "attention_mask",
            "response_mask",
            "old_logprobs",
            "ref_logprobs",
        ]:
            if key in train_batch and torch.is_tensor(train_batch[key]):
                train_batch[key] = train_batch[key].to(self.device)

        return train_batch

    def compute_loss(self, train_batch):
        input_ids = train_batch["input_ids"]
        attention_mask = train_batch["attention_mask"]
        response_mask = train_batch["response_mask"]
        old_logprobs = train_batch["old_logprobs"]
        advantages = train_batch["advantages"]

        new_logprobs = get_response_logprobs(self.actor, input_ids, attention_mask=attention_mask)
        log_ratio = new_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        advantages = advantages.unsqueeze(-1)  # [B, 1]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2)

        # Mask out non-response tokens
        pg_loss = masked_mean(policy_loss, response_mask, dim=1)
        pg_loss = pg_loss.mean()

        metrics = {
            "reward_mean": train_batch["rewards"].mean().item(),
            "reward_std": train_batch["rewards"].std(unbiased=False).item(),
            "adv_mean": train_batch["advantages"].mean().item(),
            "adv_std": train_batch["advantages"].std(unbiased=False).item(),
            "ratio_mean": masked_mean(ratio, response_mask, dim=1).mean().item(),
            "logprob_mean": masked_mean(new_logprobs, response_mask, dim=1).mean().item(),
            "policy_loss": pg_loss.item(),
        }

        loss = pg_loss

        # KL penalty
        if self.kl_coef > 0.0 and "ref_logprobs" in train_batch:
            ref_logprobs = train_batch["ref_logprobs"]
            kl_loss = masked_mean(new_logprobs - ref_logprobs, response_mask, dim=1).mean()
            loss += self.kl_coef * kl_loss
            metrics["kl_loss"] = kl_loss.item()

        metrics["total_loss"] = loss.item()
        return loss, metrics

    def _compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.normalize_advantages:
            return rewards

        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, unbiased=False, keepdim=True).clamp_min(self.eps)
        advantages = (rewards - mean) / std
        return advantages
