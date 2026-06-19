import contextlib
import csv
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path


def configure_thread_env(num_threads):
    if num_threads is None:
        return

    value = str(num_threads)

    thread_vars = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]

    for var in thread_vars:
        os.environ[var] = value

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def apply_library_thread_limits(num_threads):
    if num_threads is None:
        return

    try:
        import cv2
        cv2.setNumThreads(num_threads)
    except Exception:
        pass

    try:
        import torch
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, min(num_threads, 4)))
    except Exception:
        pass


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def append_runtime_record(log_path, record):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = {
        key: _json_safe(value)
        for key, value in record.items()
    }

    fieldnames = [
        "timestamp",
        "command",
        "status",
        "elapsed_seconds",
        "num_threads",
        "python_version",
        "platform",
        "metadata_json",
        "error",
    ]

    write_header = not log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(cleaned)


@contextlib.contextmanager
def timed_command(command, log_path=None, num_threads=None, metadata=None):
    start = time.perf_counter()
    status = "ok"
    error = ""

    try:
        yield
    except Exception as exc:
        status = "error"
        error = repr(exc)
        raise
    finally:
        elapsed = time.perf_counter() - start

        if log_path is not None:
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "command": command,
                "status": status,
                "elapsed_seconds": round(elapsed, 6),
                "num_threads": num_threads,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "metadata_json": json.dumps(metadata or {}),
                "error": error,
            }

            append_runtime_record(log_path, record)

        print(f"Runtime: {elapsed:.3f} seconds")