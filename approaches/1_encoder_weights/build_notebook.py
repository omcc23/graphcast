"""Build a self-contained Kaggle notebook for the weight-energy analysis."""

import base64
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


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
    mesh = (ROOT / "icosahedral_mesh.py").read_bytes()
    analysis = (ROOT / "weight_entropy.py").read_bytes()
    payload = {
        "icosahedral_mesh.py": base64.b64encode(mesh).decode("ascii"),
        "weight_entropy.py": base64.b64encode(analysis).decode("ascii"),
    }
    markdown = """# GraphCast weight entropy on the real mesh lengths

The published checkpoints are not retrained. Each one stores a single mesh-edge encoder and applies it to every multimesh edge, so there is no separate weight matrix for M0, M1, and so on.

The energy at level m is the mean squared output of those shared weights on the edges whose length is L(m). L(m) = pi R / (2 * 2^m), with R = 6371 km, and the wavenumber is k = 1/L. The hierarchy stops at `model_config.mesh_size`: M0-M5 for GraphCast Small, M0-M6 for the operational and full checkpoints. The 50 km, 25 km, and 12 km rungs are not edges of these checkpoints, so they are not given an energy.

Three outputs of the same weights are reported, in nats:

- `linear_0`: the first matrix only, which is the quantity in `Main codes`.
- `mlp`: both linear maps, with swish between them, before layer norm.
- `encoder`: that result after the trained layer norm, which is the edge embedding the encoder writes.

H = -sum p_i ln p_i and H_n = H / ln N.
"""
    runner = (
        "import base64\n"
        "from pathlib import Path\n"
        f"PAYLOAD = {payload!r}\n"
        "root = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path('.')\n"
        "for name, encoded in PAYLOAD.items():\n"
        "    path = root / name\n"
        "    path.write_bytes(base64.b64decode(encoded))\n"
        "    print('wrote', path)\n"
        "import runpy\n"
        "runpy.run_path(str(root / 'weight_entropy.py'), run_name='__main__')\n"
    )
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
        "cells": [
            cell("markdown", markdown),
            cell("code", runner),
        ],
    }
    destination = ROOT / "graphcast_latent_entropy.ipynb"
    destination.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
