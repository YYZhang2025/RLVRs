from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvrs.algs.grpo import GRPOTrainer
from rlvrs.rollout import RolloutBackend, build_rollout_engine
from rlvrs.verifiers.gsm8k_verifier import GSM8KVerifier

SYSTEM_PROMPT = (
    "You are a careful math reasoning assistant.\n"
    "For every problem, respond in exactly this format:\n"
    "<think>\n"
    "step-by-step reasoning\n"
    "</think>\n"
    "\\boxed{final_answer}\n"
    "Do not omit the <think> tags.\n"
    "Do not output anything after the boxed final answer."
)


def format_prompt(question: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        "Return your answer using this exact structure:\n"
        "<think>\n"
        "...\n"
        "</think>\n"
        "\\boxed{...}"
    )


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"

    lr: float = 1e-6
    weight_decay: float = 0.0
    batch_size: int = 2
    group_size: int = 4
    max_prompt_length: int = 512
    max_new_tokens: int = 256
    num_steps: int = 50
    grad_accum_steps: int = 1

    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20

    clip_range: float = 0.2
    reward_scale: float = 1.0
    kl_coef: float = 0.0
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True

    use_chat_template: bool = True
    add_generation_prompt: bool = True


class SimpleListDataLoader:
    def __init__(self, rows: List[Dict], batch_size: int) -> None:
        self.rows = rows
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.rows), self.batch_size):
            chunk = self.rows[i : i + self.batch_size]
            yield {
                "prompts": [x["prompt"] for x in chunk],
                "answers": [x["answer"] for x in chunk],
            }


def prepare_dataset(cfg: TrainConfig):
    # ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for ex in ds:
        rows.append(
            {
                "prompt": format_prompt(ex["question"]),
                "answer": ex["answer"],
            }
        )
    return rows


def main():
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    rollout_engine = build_rollout_engine(
        backend=RolloutBackend.HUGGINGFACE,
        model=model,
        tokenizer=tokenizer,
        group_size=cfg.group_size,
        max_prompt_length=cfg.max_prompt_length,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        device=device,
        use_chat_template=cfg.use_chat_template,
        add_generation_prompt=cfg.add_generation_prompt,
    )

    verifier = GSM8KVerifier(device=device)

    trainer = GRPOTrainer(
        actor=model,
        verifier=verifier,
        optimizer=optimizer,
        rollout_engine=rollout_engine,
        device=device,
        config={
            "clip_range": cfg.clip_range,
            "reward_scale": cfg.reward_scale,
            "kl_coef": cfg.kl_coef,
            "max_grad_norm": cfg.max_grad_norm,
            "grad_accum_steps": cfg.grad_accum_steps,
            "normalize_advantages": cfg.normalize_advantages,
            "group_size": cfg.group_size,
            "use_mixed_precision": torch.cuda.is_available(),
        },
    )

    rows = prepare_dataset(cfg)
    dataloader = SimpleListDataLoader(rows, batch_size=cfg.batch_size)

    step = 0
    for batch in dataloader:
        metrics = trainer.train_step(batch)

        if step % 5 == 0:
            print(
                f"step={step} "
                f"loss={metrics['loss']:.4f} "
                f"reward_mean={metrics.get('reward_mean', 0.0):.4f} "
                f"ratio_mean={metrics.get('ratio_mean', 0.0):.4f} "
                f"optimizer_step={metrics.get('optimizer_step', False)}"
            )

        step += 1
        if step >= cfg.num_steps:
            break

    flush_metrics = trainer.flush()
    if flush_metrics is not None:
        print("flushed:", flush_metrics)

    save_dir = "outputs/qwen3_5_0p8b_grpo_gsm8k"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"saved to {save_dir}")


if __name__ == "__main__":
    main()
