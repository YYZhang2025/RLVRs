from __future__ import annotations

import re
from typing import Optional

import torch

from rlvrs.schema import RolloutBatch

from .base import BaseVerifier

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_GSM8K_FINAL_RE = re.compile(r"####\s*([^\n]+)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    text = re.sub(r"[\s\.\,;:!?\)\]\}]+$", "", text)
    text = re.sub(r"^[\s\(\[\{]+", "", text)
    return text.strip()


def extract_gold_answer(answer_text: str) -> str:
    m = _GSM8K_FINAL_RE.search(answer_text)
    if m:
        return _normalize_number(m.group(1))

    nums = _NUMBER_RE.findall(answer_text)
    if nums:
        return _normalize_number(nums[-1])

    return _normalize_number(answer_text)


def extract_pred_answer(response: str) -> str:
    m = _BOXED_RE.search(response)
    if m:
        return _normalize_number(m.group(1))

    nums = _NUMBER_RE.findall(response)
    if nums:
        return _normalize_number(nums[-1])

    return _normalize_number(response)


class GSM8KVerifier(BaseVerifier):
    """
    Expects rollout_batch.extra["answers"] = List[str] of raw GSM8K answers.
    Reward:
        1.0 if extracted final answer matches gold final answer
        0.0 otherwise
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__(device=device)

    def compute_rewards(self, rollout_batch: RolloutBatch) -> torch.Tensor:
        if rollout_batch.extra is None or "answers" not in rollout_batch.extra:
            raise KeyError("GSM8KVerifier expects rollout_batch.extra['answers'].")

        raw_answers = rollout_batch.extra["answers"]
        if not isinstance(raw_answers, list) or not all(isinstance(x, str) for x in raw_answers):
            raise TypeError("rollout_batch.extra['answers'] must be a List[str].")

        if len(raw_answers) != rollout_batch.num_prompts:
            raise ValueError(
                f"answers length mismatch: got {len(raw_answers)}, expected {rollout_batch.num_prompts}."
            )

        gold_answers = [extract_gold_answer(x) for x in raw_answers]

        expanded_gold = []
        for ans in gold_answers:
            expanded_gold.extend([ans] * rollout_batch.group_size)

        rewards = []
        pred_answers = []
        for response, gold in zip(rollout_batch.responses, expanded_gold):
            pred = extract_pred_answer(response)
            pred_answers.append(pred)
            rewards.append(1.0 if pred == gold else 0.0)

        if rollout_batch.extra is None:
            rollout_batch.extra = {}
        rollout_batch.extra["gold_answers"] = gold_answers
        rollout_batch.extra["pred_answers"] = pred_answers

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rlvrs.rollout.huggingface import HuggingFaceRolloutEngine
    from rlvrs.verifiers.exact_match import ExactMatchVerifier

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    engine = HuggingFaceRolloutEngine(
        model=model,
        tokenizer=tokenizer,
        group_size=2,
        max_prompt_length=128,
        max_new_tokens=16,
        do_sample=False,
        use_chat_template=True,
        add_generation_prompt=True,
        device=device,
    )

    verifier = GSM8KVerifier(device=device)

    SYSTEM_PROMPT = (
        "You are a careful math reasoning assistant.\n"
        "First think inside <think> and </think> tags.\n"
        "Then give the final answer.\n"
        "For this task, output only one numeral as the final answer."
        "For example, if the question is 'What is 1 + 1?', a good response would be 'The answer is <think>1 + 1 = 2</think> 2'."
    )

    batch = {
        "prompts": [
            f"{SYSTEM_PROMPT}\n\nQuestion: What is 1 + 1?",
            f"{SYSTEM_PROMPT}\n\nQuestion: What is 4 - 2?",
        ],
        "answers": [
            "2",
            "2",
        ],
    }

    rollout_batch = engine.rollout(batch)
    if rollout_batch.extra is None:
        rollout_batch.extra = {}
    rollout_batch.extra["answers"] = batch["answers"]

    scored_batch = verifier.score(rollout_batch)
    print("=== Responses ===")
    for i in range(rollout_batch.num_prompts):
        print(f"Prompt {i}: {rollout_batch.prompts[i]}")
        print(f"Response {i}: {rollout_batch.responses[i]}")
        print(f"Gold Answer {i}: {scored_batch.extra['gold_answers'][i]}")
        print(f"Pred Answer {i}: {scored_batch.extra['pred_answers'][i]}")
        print(f"Reward {i}: {scored_batch.rewards[i].item()}")
        print("-------------")
