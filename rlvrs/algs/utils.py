from typing import Optional

import torch
import torch.nn as nn


def get_response_logprobs(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs
):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **kwargs,
    )

    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_logprobs = torch.gather(logprobs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    return target_logprobs


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
):
    mask = mask.float()
    denom = mask.sum(dim=dim).clamp_min(1) if dim is not None else mask.sum().clamp_min(1)
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / denom if dim is not None else masked_tensor.sum() / denom
