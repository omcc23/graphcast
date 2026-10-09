"""Build the Kaggle notebook that measures processor-edge energy."""

import base64
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "approach3"


def cell(kind: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    item = {
        "cell_type": kind,
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": lines,
    }
    if kind == "code":
        item["outputs"] = []
        item["execution_count"] = None
    return item


def main() -> None:
    script = (ROOT / "processor_edge_entropy.py").read_bytes()
    payload = base64.b64encode(script).decode("ascii")
    markdown = """# GraphCast processor edge entropy

GraphCast Small, one 6-hour step on the published ERA5 state for 2022-01-01. Nothing is retrained.

Each processor step writes a 512-vector on every multimesh edge, before the residual is added. The energy of a mesh level is the mean squared norm of those vectors. The row labeled encoder is the mesh-edge embedding produced before the processor.
"""
    runner = f"""import base64
import os
import subprocess
import sys
from pathlib import Path

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--upgrade",
    "git+https://github.com/google-deepmind/weathernext.git@v0.3.0",
])
root = Path("/kaggle/working")
script = root / "processor_edge_entropy.py"
script.write_bytes(base64.b64decode({payload!r}))
print("running", script, "in a new interpreter", flush=True)
env = os.environ.copy()
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
env["PYTHONUNBUFFERED"] = "1"
subprocess.check_call([sys.executable, str(script)], env=env)
"""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [cell("markdown", markdown), cell("code", runner)],
    }
    DEST.mkdir(parents=True, exist_ok=True)
    destination = DEST / "graphcast_processor_edge_entropy.ipynb"
    destination.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    metadata = {
        "id": "shreyasgupta13/graphcast-processor-edge-entropy",
        "title": "GraphCast processor edge entropy",
        "code_file": "graphcast_processor_edge_entropy.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (DEST / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
