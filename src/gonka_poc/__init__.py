"""gonka-poc: out-of-tree vLLM plugin for Gonka Proof-of-Compute (PoC v2).

This package ships as a standalone pip-installable plugin that targets a vllm
build carrying the PoC engine seams (0.25.x). It provides two integration
surfaces:

1. ``vllm.general_plugins`` entry point (:func:`gonka_poc.plugin.register`)
   that adopts vLLM's logging and installs the EngineCore KV borrow/return
   utility methods for leased-block validation (ADR-0015).
2. ``--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`` exposing
   ``execute_poc_decode`` to vLLM's ``collective_rpc``.

The PoC HTTP routes are mounted by ``gonka-vllm-serve`` when stock
``build_app`` has no ``/api/v1/pow`` paths. ``vllm serve`` on an unpatched
MetaX tree does not register them.

The fork-residual changes (sampler-stack + structured-output) are NOT shipped
here -- see ``MIGRATION_FROM_FORK.md`` for the disposition of every commit
from the source branch.
"""

# Single source of truth for the version is pyproject.toml; derive it so a
# version bump cannot leave a stale literal behind.
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("gonka-poc")
except Exception:  # pragma: no cover - not installed (e.g. source checkout)
    __version__ = "unknown"
