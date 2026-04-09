from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from rlvrs.schema import RolloutBatch


class HuggingFaceRolloutEngine:
    """
    Local rollout engine for debugging based on Hugging Face generate().

    Input batch format:
        {
            "prompts": List[str]
        }
    or
        {
            "texts": List[str]
        }

    Output:
        RolloutBatch

    Notes:
        - batch_size after rollout = num_prompts * group_size
        - old_logprobs are recomputed from the rollout policy on final sequences
        - response_mask marks generated tokens only, shape [B, T]
        - prompts stored in RolloutBatch.prompts are the raw user prompts, not formatted chat prompts
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        group_size: int = 8,
        max_prompt_length: int = 1024,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        device: Optional[torch.device] = None,
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.use_chat_template = use_chat_template
        self.add_generation_prompt = add_generation_prompt

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = device

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must have either pad_token_id or eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def rollout(self, batch: Dict[str, Any]) -> RolloutBatch:
        raw_prompts = self._get_prompts(batch)
        num_prompts = len(raw_prompts)
        if num_prompts == 0:
            raise ValueError("No prompts to rollout.")

        prompt_enc, formatted_prompts = self._encode_prompts(raw_prompts)

        prompt_input_ids = prompt_enc["input_ids"].to(self.device)  # [N, P]
        prompt_attention_mask = prompt_enc["attention_mask"].to(self.device)  # [N, P]

        repeated_prompt_input_ids = self._repeat_interleave_rows(prompt_input_ids, self.group_size)  # [B, P]
        repeated_prompt_attention_mask = self._repeat_interleave_rows(
            prompt_attention_mask, self.group_size
        )  # [B, P]

        repeated_raw_prompts = self._repeat_text(raw_prompts, self.group_size)  # len=B
        repeated_formatted_prompts = self._repeat_text(formatted_prompts, self.group_size)  # len=B

        self.model.eval()

        generate_kwargs = {
            "input_ids": repeated_prompt_input_ids,
            "attention_mask": repeated_prompt_attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
            "repetition_penalty": self.repetition_penalty,
        }

        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                generate_kwargs["top_k"] = self.top_k

        generated = self.model.generate(**generate_kwargs)
        input_ids = generated.sequences  # [B, T]

        attention_mask = self._build_attention_mask(input_ids)  # [B, T]

        prompt_lengths = repeated_prompt_attention_mask.sum(dim=-1)  # [B]
        full_lengths = attention_mask.sum(dim=-1)  # [B]

        response_mask = self._build_response_mask(
            seq_len=input_ids.size(1),
            prompt_lengths=prompt_lengths,
            full_lengths=full_lengths,
            device=input_ids.device,
        )  # [B, T]

        old_logprobs = self._compute_old_logprobs(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # [B, T-1]

        responses = self._decode_responses_only(
            full_input_ids=input_ids,
            prompt_lengths=prompt_lengths,
        )

        extra: Dict[str, Any] = {
            "prompt_lengths": prompt_lengths,
            "full_lengths": full_lengths,
            "formatted_prompts": repeated_formatted_prompts,
            "used_chat_template": self._should_use_chat_template(),
        }

        return RolloutBatch(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_mask=response_mask,
            old_logprobs=old_logprobs,
            responses=responses,
            prompts=repeated_raw_prompts,
            group_size=self.group_size,
            num_prompts=num_prompts,
            ref_logprobs=None,
            extra=extra,
        )

    def _get_prompts(self, batch: Dict[str, Any]) -> List[str]:
        if "prompts" in batch:
            prompts = batch["prompts"]
        elif "texts" in batch:
            prompts = batch["texts"]
        else:
            raise KeyError("batch must contain 'prompts' or 'texts'.")

        if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
            raise TypeError("'prompts' must be a List[str].")

        return prompts

    def _should_use_chat_template(self) -> bool:
        return self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template")

    def _encode_prompts(
        self,
        prompts: List[str],
    ) -> tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Returns:
            prompt_enc: tokenizer output with input_ids / attention_mask
            formatted_prompts: strings actually used before tokenization, for debugging
        """
        if self._should_use_chat_template():
            try:
                messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]

                prompt_enc = self.tokenizer.apply_chat_template(
                    messages_batch,
                    tokenize=True,
                    add_generation_prompt=self.add_generation_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length,
                    return_dict=True,
                )

                formatted_prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=self.add_generation_prompt,
                    )
                    for prompt in prompts
                ]
                return prompt_enc, formatted_prompts

            except Exception:
                # fallback to string formatting + normal tokenization
                formatted_prompts = [self._format_prompt_with_chat_template(prompt) for prompt in prompts]
                prompt_enc = self.tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length,
                )
                return prompt_enc, formatted_prompts

        formatted_prompts = prompts
        prompt_enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        return prompt_enc, formatted_prompts

    def _format_prompt_with_chat_template(self, prompt: str) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )

        # fallback for templates that fail to append assistant start
        if self.add_generation_prompt and "<|im_start|>assistant" not in text:
            if text.endswith("\n"):
                text = text + "<|im_start|>assistant\n"
            else:
                text = text + "\n<|im_start|>assistant\n"

        return text

    def _repeat_interleave_rows(self, x: torch.Tensor, repeats: int) -> torch.Tensor:
        return torch.repeat_interleave(x, repeats=repeats, dim=0)

    def _repeat_text(self, texts: List[str], repeats: int) -> List[str]:
        out: List[str] = []
        for text in texts:
            out.extend([text] * repeats)
        return out

    def _build_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Build attention mask from padded sequences.

        For most decoder-only models in debugging mode, this is sufficient.
        """
        return (input_ids != self.tokenizer.pad_token_id).long()

    def _build_response_mask(
        self,
        seq_len: int,
        prompt_lengths: torch.Tensor,
        full_lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build response mask of shape [B, T].

        For each sample:
            positions in [prompt_len, full_len) are 1
            others are 0
        """
        batch_size = prompt_lengths.size(0)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        start = prompt_lengths.unsqueeze(1)
        end = full_lengths.unsqueeze(1)
        response_mask = ((positions >= start) & (positions < end)).long()
        return response_mask

    @torch.inference_mode()
    def _compute_old_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token logprobs for labels=input_ids[:, 1:].

        Returns:
            old_logprobs: [B, T-1]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs  # [B, T, V]

        shift_logits = logits[:, :-1, :]  # [B, T-1, V]
        shift_labels = input_ids[:, 1:]  # [B, T-1]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)  # [B, T-1]

        return token_logprobs

    def _decode_responses_only(
        self,
        full_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> List[str]:
        responses: List[str] = []
        for seq, prompt_len in zip(full_input_ids, prompt_lengths):
            response_ids = seq[int(prompt_len.item()) :]
            text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(text)
        return responses


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    engine = HuggingFaceRolloutEngine(
        model=model,
        tokenizer=tokenizer,
        group_size=3,
        max_prompt_length=128,
        max_new_tokens=16,
        do_sample=False,
        use_chat_template=True,
        add_generation_prompt=True,
        device=device,
    )

    batch = {
        "prompts": [
            "What is the LLM?",
            "How to use RL to train LLMs?",
        ]
    }

    rollout_batch = engine.rollout(batch)

    print("input_ids:", rollout_batch.input_ids.shape)
    print("attention_mask:", rollout_batch.attention_mask.shape)
    print("response_mask:", rollout_batch.response_mask.shape)
    print("old_logprobs:", rollout_batch.old_logprobs.shape)
    print("used_chat_template:", rollout_batch.extra["used_chat_template"])

    for i in range(rollout_batch.batch_size):
        print(f"\n====== Sample {i} ======")
        print("Prompt:", rollout_batch.prompts[i])
        print("Formatted Prompt:", rollout_batch.extra["formatted_prompts"][i])
        print("Response:", rollout_batch.responses[i])
