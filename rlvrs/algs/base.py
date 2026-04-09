from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

TensorDict = Dict[str, Any]


class BaseTrainer(ABC):
    def __init__(
        self,
        actor: nn.Module,
        verifier: Any,
        optimizer: optim.Optimizer,
        config: Optional[Dict[str, Any]] = None,
        rollout_engine: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.actor = actor
        self.verifier = verifier
        self.optimizer = optimizer
        self.config = config or {}
        self.rollout_engine = rollout_engine
        self.lr_scheduler = lr_scheduler

        if device is None:
            try:
                self.device = next(actor.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.global_step = 0  # number of micro-steps
        self.update_step = 0  # number of optimizer steps

        self.max_grad_norm = self.config.get("max_grad_norm", None)
        self.use_mixed_precision = self.config.get("use_mixed_precision", False)
        self.grad_accum_steps = int(self.config.get("grad_accum_steps", 1))
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")

        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)
        self.scaler = None

        # track where we are inside the current accumulation cycle
        self._accum_step = 0

        # zero grad once at startup
        self.optimizer.zero_grad(set_to_none=True)

    def train(self) -> None:
        self.actor.train()

    def eval(self) -> None:
        self.actor.eval()

    def train_step(self, batch: TensorDict) -> Dict[str, Any]:
        """
        One micro-step.

        Returns metrics for this micro-step.
        `optimizer_step=True` means parameters were updated this call.
        """
        self.train()

        rollout_batch = self.rollout(batch)
        scored_batch = self.score(rollout_batch)
        train_batch = self.build_train_batch(scored_batch)

        loss, metrics = self._compute_loss_and_metrics(train_batch)

        # divide loss for gradient accumulation
        scaled_loss = loss / self.grad_accum_steps

        self.backward(scaled_loss)

        self._accum_step += 1
        self.global_step += 1

        optimizer_step_happened = False
        opt_metrics: Dict[str, Any] = {}

        if self._should_optimizer_step():
            opt_metrics = self.optimizer_step()
            self._accum_step = 0
            self.update_step += 1
            optimizer_step_happened = True

        merged_metrics = {
            "loss": float(loss.detach().item()),  # report original loss, not divided one
            "global_step": self.global_step,
            "update_step": self.update_step,
            "grad_accum_step": self._accum_step if not optimizer_step_happened else self.grad_accum_steps,
            "grad_accum_steps": self.grad_accum_steps,
            "optimizer_step": optimizer_step_happened,
            **metrics,
            **opt_metrics,
        }
        return merged_metrics

    def _should_optimizer_step(self) -> bool:
        return self._accum_step >= self.grad_accum_steps

    def backward(self, loss: torch.Tensor) -> None:
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self) -> Dict[str, Any]:
        """
        Perform the real optimizer update after accumulation is complete.
        """
        grad_norm = None

        if self.use_mixed_precision and self.scaler is not None:
            if self.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)

        lr = self.optimizer.param_groups[0]["lr"]
        metrics: Dict[str, Any] = {"lr": float(lr)}

        if grad_norm is not None:
            metrics["grad_norm"] = (
                float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
            )

        return metrics

    def flush(self) -> Optional[Dict[str, Any]]:
        """
        Force an optimizer step if there are leftover accumulated gradients.
        Useful at the end of an epoch when number of micro-batches is not divisible
        by grad_accum_steps.
        """
        if self._accum_step == 0:
            return None

        opt_metrics = self.optimizer_step()
        self._accum_step = 0
        self.update_step += 1

        return {
            "global_step": self.global_step,
            "update_step": self.update_step,
            "optimizer_step": True,
            "flushed": True,
            **opt_metrics,
        }

    def rollout(self, batch: TensorDict) -> TensorDict:
        if self.rollout_engine is None:
            raise NotImplementedError(
                "Rollout engine is not defined. Please provide rollout_engine or override rollout()."
            )
        return self.rollout_engine.rollout(batch)

    def score(self, rollout_batch: TensorDict) -> TensorDict:
        if self.verifier is None:
            raise NotImplementedError("Verifier is not defined. Please provide verifier or override score().")

        if hasattr(self.verifier, "score"):
            scores = self.verifier.score(rollout_batch)
        elif callable(self.verifier):
            scores = self.verifier(rollout_batch)
        else:
            raise TypeError("Verifier must implement `.score(...)` or be callable.")

        merged = dict(rollout_batch)

        if isinstance(scores, dict):
            merged.update(scores)
            return merged

        if torch.is_tensor(scores):
            merged["rewards"] = scores
            return merged

        raise TypeError("Verifier output must be either dict or torch.Tensor.")

    @abstractmethod
    def build_train_batch(self, scored_batch: TensorDict) -> TensorDict:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, train_batch: TensorDict) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def _compute_loss_and_metrics(self, train_batch: TensorDict) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                loss, metrics = self.compute_loss(train_batch)
        else:
            loss, metrics = self.compute_loss(train_batch)

        if not torch.is_tensor(loss):
            raise TypeError("compute_loss must return a torch.Tensor as loss.")
        if loss.dim() != 0:
            raise ValueError("Loss tensor must be scalar.")
        if not isinstance(metrics, dict):
            raise TypeError("compute_loss must return metrics as a dict.")

        return loss, metrics

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_step": self.update_step,
            "config": self.config,
            "_accum_step": self._accum_step,
        }
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.use_mixed_precision and self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.global_step = state_dict.get("global_step", 0)
        self.update_step = state_dict.get("update_step", 0)
        self._accum_step = state_dict.get("_accum_step", 0)

        if self.lr_scheduler is not None and "lr_scheduler" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        if self.use_mixed_precision and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])
