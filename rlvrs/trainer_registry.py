from __future__ import annotations

from typing import Any, Callable, Dict, Type

TRAINER_REGISTRY: Dict[str, Type] = {}


def register_trainer(name: str) -> Callable[[Type], Type]:
    name = name.strip().lower()

    def decorator(cls: Type) -> Type:
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


def get_trainer(name: str) -> Type:
    name = name.strip().lower()
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Trainer '{name}' is not registered.")
    return TRAINER_REGISTRY[name]


def create_trainer(name: str, *args: Any, **kwargs: Any) -> Any:
    trainer_cls = get_trainer(name)
    return trainer_cls(*args, **kwargs)


def create_trainer_from_config(config: Dict[str, Any]) -> Any:
    if "trainer_name" not in config:
        raise ValueError("Config must contain 'trainer_name' key.")
    trainer_name = config["trainer_name"]
    trainer_cls = get_trainer(trainer_name)
    return trainer_cls(**config)


def list_trainer_names() -> list[str]:
    return list(TRAINER_REGISTRY.keys())
