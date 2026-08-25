# SPDX-License-Identifier: CC-BY-SA-4.0

import atexit
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
try:
    from IPython import get_ipython
except Exception:
    get_ipython = None


Bundle = Dict[str, Any]
BundleFactory = Callable[[], Bundle]
CanAttachFn = Callable[[List[str], str, Any, Any], bool]
ResetFn = Callable[[], None]
SummaryFn = Callable[[Bundle, bool, int, bool], None]


@dataclass
class _ActiveRun:
    task_key: Any
    stages: List[str]
    bundle: Bundle
    reset_cb: ResetFn
    summary_cb: SummaryFn
    quiet: bool
    display_decimals: int
    display_truncate: bool


_LOCK = threading.RLock()
_ACTIVE_RUN: Optional[_ActiveRun] = None
_HOOKS_INSTALLED: bool = False


def _stamp_bundle(bundle: Bundle, stages: List[str], is_open: bool) -> None:
    """
    Keep minimal run metadata inside the shared results bundle.

    This is informational only. Lifecycle is owned by this module.
    """
    meta = bundle.setdefault("metadata", {})
    run_meta = meta.setdefault("run", {})
    run_meta["stages"] = list(stages)
    run_meta["open"] = bool(is_open)


def _finalise_open_run_locked() -> None:
    """
    Finalise the currently open run, if any.

    Summary printing is optional via `quiet`, but reset is unconditional.
    """
    global _ACTIVE_RUN
    run = _ACTIVE_RUN
    if run is None:
        return

    _ACTIVE_RUN = None
    _stamp_bundle(run.bundle, run.stages, is_open=False)

    run_meta = run.bundle.setdefault("metadata", {}).setdefault("run", {})
    run_meta["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    started = run_meta.pop("_started_monotonic", None)
    if started is not None:
        run_meta["elapsed_seconds"] = round(time.monotonic() - started, 3)

    try:
        run.summary_cb(
            run.bundle,
            run.quiet,
            run.display_decimals,
            run.display_truncate,
        )
    finally:
        run.reset_cb()


def finalise_open_run(*_args, **_kwargs) -> None:
    """
    Public finaliser.

    Accepts arbitrary args so it can be safely registered as an IPython
    `post_run_cell` callback as well as an `atexit` handler.
    """
    with _LOCK:
        _finalise_open_run_locked()


def get_active_run_checkpoint_timestamp() -> str:
    """
    Return the checkpoint timestamp owned by the currently active run.

    If the run has not saved anything yet, create the timestamp once and
    store it in the run metadata.
    """
    with _LOCK:
        run = _ACTIVE_RUN

        meta = run.bundle.setdefault("metadata", {})
        run_meta = meta.setdefault("run", {})
        timestamp = run_meta.get("checkpoint_run_timestamp")

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_meta["checkpoint_run_timestamp"] = timestamp

        return str(timestamp)


def install_boundary_hooks() -> None:
    """
    Install exactly-once lifecycle hooks.

    - In scripts / HPC jobs: finalise at process exit.
    - In IPython / Jupyter / Colab: finalise at the end of each cell.

    Treating a notebook cell as a script matches the intended run semantics.
    """
    global _HOOKS_INSTALLED

    with _LOCK:
        if _HOOKS_INSTALLED:
            return

        atexit.register(finalise_open_run)

        ip = get_ipython() if callable(get_ipython) else None
        if ip is not None and hasattr(ip, "events"):
            try:
                ip.events.register("post_run_cell", finalise_open_run)
            except Exception:
                pass

        _HOOKS_INSTALLED = True


def begin_or_attach_run(
    *,
    task_key: Any,
    stage: str,
    bundle_factory: BundleFactory,
    can_attach: CanAttachFn,
    reset_cb: ResetFn,
    summary_cb: SummaryFn,
    quiet: bool,
    display_decimals: int,
    display_truncate: bool,
) -> Bundle:
    """
    Start a new run or attach to the currently open one.

    Expected policy from the caller:
      - First entrypoint in a script/cell opens a run.
      - A same-task `tnn -> gnn_full` or `tnn -> gnn_edges` sequence can attach
        to that same run.
      - Any other top-level entrypoint starts a new run, which first finalises
        the old one.

    Parameters
    ----------
    task_key:
        Stable identity for the task inside the current process. `id(task)` is
        sufficient for the current pipeline usage.
    stage:
        One of the caller-defined stage names, e.g. `"tnn"`, `"gnn_full"`,
        `"gnn_edges"`.
    bundle_factory:
        Zero-arg callable returning a fresh results bundle.
    can_attach:
        Predicate controlling whether the incoming stage should attach to the
        active run. Signature:
            (active_stages, next_stage, active_task_key, next_task_key) -> bool
    reset_cb:
        Callback that clears pipeline-global state such as feature registries,
        caches, and warning trackers.
    summary_cb:
        Callback that optionally prints the final summary table for a completed
        run. It must respect `quiet`, but lifecycle cleanup must not depend on
        it.
    quiet:
        Suppresses summary printing only.
    display_decimals / display_truncate:
        Display options forwarded to `summary_cb`.

    Returns
    -------
    dict
        The active run bundle to be populated by the caller.
    """
    global _ACTIVE_RUN
    install_boundary_hooks()

    with _LOCK:
        if _ACTIVE_RUN is not None:
            active = _ACTIVE_RUN

            if can_attach(list(active.stages), stage, active.task_key, task_key):
                if stage not in active.stages:
                    active.stages.append(stage)

                # Latest caller controls final summary display behavior.
                active.quiet = bool(quiet)
                active.display_decimals = int(display_decimals)
                active.display_truncate = bool(display_truncate)

                # If some caller printed early by mistake, ensure end-of-run
                # finalisation is still allowed to print once.
                active.bundle.pop("_summary_printed", None)
                _stamp_bundle(active.bundle, active.stages, is_open=True)
                return active.bundle

            # New top-level run in the same cell/script => close the previous one.
            _finalise_open_run_locked()

        bundle = bundle_factory()
        if not isinstance(bundle, dict):
            raise TypeError("bundle_factory() must return a dict-like bundle.")

        bundle.setdefault("results", {})
        bundle.setdefault("metadata", {})
        bundle.pop("_summary_printed", None)

        _ACTIVE_RUN = _ActiveRun(
            task_key=task_key,
            stages=[stage],
            bundle=bundle,
            reset_cb=reset_cb,
            summary_cb=summary_cb,
            quiet=bool(quiet),
            display_decimals=int(display_decimals),
            display_truncate=bool(display_truncate),
        )
        _stamp_bundle(bundle, _ACTIVE_RUN.stages, is_open=True)
        run_meta = bundle["metadata"].setdefault("run", {})
        run_meta["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_meta["_started_monotonic"] = time.monotonic()
        return bundle
