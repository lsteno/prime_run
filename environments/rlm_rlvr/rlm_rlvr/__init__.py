from __future__ import annotations

from typing import Any


def load_environment(*args: Any, **kwargs: Any):
    from .env import load_environment as _load_environment

    return _load_environment(*args, **kwargs)


__all__ = ["load_environment"]
