"""OpRegistry, global config state, and the ``dispatch`` helper.

``OpRegistry`` holds, for each op name, the pure-PyTorch impl and (optionally) a
kernel adapter. ``dispatch`` looks at the current ``DispatchConfig`` and picks
one of the two.

Adapter contract (see ``kernel_adapters.py``):
    * Same positional and keyword args as the torch impl.
    * Extra trailing keyword-only ``_config: DispatchConfig`` that the dispatcher
      forwards, so adapters can read ``kernel_simulator`` etc.
    * Return value must match the torch impl's shape and dtype so the swap is
      transparent to callers of ``ops.*``.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Callable, Dict, Iterator, Optional, Tuple

from .config import DispatchConfig


class OpRegistry:
    """Name -> (torch impl, optional kernel adapter)."""

    def __init__(self) -> None:
        self._torch: Dict[str, Callable] = {}
        self._kernel: Dict[str, Callable] = {}

    def register_torch(self, op: str, fn: Callable) -> None:
        if op in self._torch:
            raise ValueError(f"torch impl for {op!r} already registered")
        self._torch[op] = fn

    def register_kernel(self, op: str, fn: Callable) -> None:
        if op in self._kernel:
            raise ValueError(f"kernel adapter for {op!r} already registered")
        self._kernel[op] = fn

    def has_torch(self, op: str) -> bool:
        return op in self._torch

    def has_kernel(self, op: str) -> bool:
        return op in self._kernel

    def torch_impl(self, op: str) -> Callable:
        if op not in self._torch:
            raise KeyError(f"no torch impl registered for {op!r}")
        return self._torch[op]

    def kernel_impl(self, op: str) -> Callable:
        if op not in self._kernel:
            raise KeyError(f"no kernel adapter registered for {op!r}")
        return self._kernel[op]

    def summary(self) -> Dict[str, Tuple[bool, bool]]:
        """{op: (has_torch, has_kernel)} for debugging."""
        names = set(self._torch) | set(self._kernel)
        return {n: (n in self._torch, n in self._kernel) for n in sorted(names)}

    def ops(self) -> Iterator[str]:
        yield from sorted(set(self._torch) | set(self._kernel))


# Single process-wide registry and current config.
_REGISTRY = OpRegistry()
_CURRENT_CONFIG: DispatchConfig = DispatchConfig()


def registry() -> OpRegistry:
    return _REGISTRY


def set_config(cfg: DispatchConfig) -> None:
    global _CURRENT_CONFIG
    _CURRENT_CONFIG = cfg


def get_config() -> DispatchConfig:
    return _CURRENT_CONFIG


@contextlib.contextmanager
def using_config(cfg: DispatchConfig):
    """Temporarily swap the global config for the duration of the ``with`` block."""
    old = _CURRENT_CONFIG
    set_config(cfg)
    try:
        yield
    finally:
        set_config(old)


def dispatch(op: str, *args, _config: Optional[DispatchConfig] = None, **kwargs):
    """Route ``op`` to either its torch impl or its kernel adapter.

    Resolution order for the effective config:
        1. Explicit per-call ``_config`` kwarg.
        2. Global config set via ``set_config``.
        3. Default ``DispatchConfig()`` (all torch).

    Uses the kernel adapter only when ``config.use_kernel_<op>`` is True AND an
    adapter is registered. Otherwise warns (if the flag is on but no adapter)
    and falls back to the torch impl.
    """
    cfg = _config if _config is not None else _CURRENT_CONFIG
    flag_name = f"use_kernel_{op}"
    want_kernel = getattr(cfg, flag_name, False)
    if want_kernel:
        if _REGISTRY.has_kernel(op):
            return _REGISTRY.kernel_impl(op)(*args, _config=cfg, **kwargs)
        warnings.warn(
            f"dispatch: {flag_name}=True but no kernel adapter is registered "
            f"for {op!r}; falling back to torch impl.",
            stacklevel=2,
        )
    return _REGISTRY.torch_impl(op)(*args, **kwargs)
