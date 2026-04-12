from enum import Enum
from typing import Any


class RolloutBackend(str, Enum):
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"


def build_rollout_engine(backend: RolloutBackend, **kwargs) -> Any:
    if backend == RolloutBackend.HUGGINGFACE:
        from rlvrs.rollout.huggingface import HuggingFaceRolloutEngine

        return HuggingFaceRolloutEngine(**kwargs)
    elif backend == RolloutBackend.VLLM:
        from rlvrs.rollout.vllm import VLLMRolloutEngine

        return VLLMRolloutEngine(**kwargs)
    else:
        raise ValueError(f"Unsupported rollout backend: {backend}")
