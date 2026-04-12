from __future__ import annotations

import re
from typing import List, Optional

import torch

from rlvrs.schema import RolloutBatch
from rlvrs.verifiers.base import BaseVerifier


class ExactMatchVerifier(BaseVerifier):
    """
    Simple exact-match verifier.

    Expects rollout_batch.extra["answers"] to exist.

    Reward:
        1.0 if normalized(response) == normalized(answer)
        0.0 otherwise
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        strip: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        first_line_only: bool = False,
    ) -> None:
        super().__init__(device=device)
        self.strip = strip
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.first_line_only = first_line_only

    def compute_rewards(self, rollout_batch: RolloutBatch) -> torch.Tensor:
        if rollout_batch.extra is None or "answers" not in rollout_batch.extra:
            raise KeyError("ExactMatchVerifier expects rollout_batch.extra['answers'] to exist.")

        answers = rollout_batch.extra["answers"]
        if not isinstance(answers, list) or not all(isinstance(x, str) for x in answers):
            raise TypeError("rollout_batch.extra['answers'] must be a List[str].")

        if len(answers) != rollout_batch.num_prompts:
            raise ValueError(
                f"answers length mismatch: got {len(answers)}, expected {rollout_batch.num_prompts}."
            )

        expanded_answers = []
        for ans in answers:
            expanded_answers.extend([ans] * rollout_batch.group_size)

        rewards = []
        for response, answer in zip(rollout_batch.responses, expanded_answers):
            pred = self._normalize(response)
            gold = self._normalize(answer)
            rewards.append(1.0 if pred == gold else 0.0)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _normalize(self, text: str) -> str:
        if self.first_line_only:
            text = text.splitlines()[0] if text.splitlines() else text

        if self.strip:
            text = text.strip()

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        text = re.sub(r"\s+", " ", text).strip()
        return text


if __name__ == "__main__":
    import torch

    from rlvrs.schema import RolloutBatch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_prompts = 2
    group_size = 3
    batch_size = num_prompts * group_size

    # Dummy tensors, only shapes matter for this verifier
    prompt_input_ids = torch.zeros((num_prompts, 5), dtype=torch.long, device=device)
    prompt_attention_mask = torch.ones((num_prompts, 5), dtype=torch.long, device=device)

    input_ids = torch.zeros((batch_size, 8), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, 8), dtype=torch.long, device=device)
    response_mask = torch.zeros((batch_size, 8), dtype=torch.long, device=device)
    old_logprobs = torch.zeros((batch_size, 7), dtype=torch.float32, device=device)

    # prompt 0 repeated 3 times, prompt 1 repeated 3 times
    prompts = [
        "What is the capital of France?",
        "What is the capital of France?",
        "What is the capital of France?",
        "What is 2 + 2?",
        "What is 2 + 2?",
        "What is 2 + 2?",
    ]

    responses = [
        "Paris",  # correct
        "paris",  # correct if lowercase=True
        "London",  # wrong
        "4",  # correct
        " 4 ",  # correct if strip=True
        "five",  # wrong
    ]

    rollout_batch = RolloutBatch(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        old_logprobs=old_logprobs,
        responses=responses,
        prompts=prompts,
        group_size=group_size,
        num_prompts=num_prompts,
        ref_logprobs=None,
        extra={
            "answers": [
                "Paris",
                "4",
            ]
        },
    )

    verifier = ExactMatchVerifier(
        device=device,
        strip=True,
        lowercase=True,
        remove_punctuation=False,
        first_line_only=False,
    )

    scored_batch = verifier.score(rollout_batch)

    print("=== Responses ===")
    for i, r in enumerate(scored_batch.responses):
        print(f"{i}: {repr(r)}")

    print("\n=== Rewards ===")
    print(scored_batch.rewards)
    print(scored_batch.rewards.tolist())

    expected = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0], device=device)
    assert torch.allclose(scored_batch.rewards, expected), (
        f"Expected {expected.tolist()}, got {scored_batch.rewards.tolist()}"
    )

    print("\nTest passed.")
