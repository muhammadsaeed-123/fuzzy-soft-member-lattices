from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


def ensure_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, dpi: int = 600) -> list[Path]:
    """Save one publication figure as vector PDF and high-resolution PNG."""
    ensure_output(output_dir)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    fig.savefig(pdf, bbox_inches="tight")
    eps = output_dir / f"{stem}.eps"
    # EPS is created from the vector PDF to preserve text and geometry.
    import subprocess
    subprocess.run(["pdftops", "-eps", str(pdf), str(eps)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png, eps]


def parse_tuple(text: str) -> tuple[int, ...]:
    value = ast.literal_eval(text)
    if not isinstance(value, tuple) or not all(isinstance(x, int) for x in value):
        raise ValueError(f"Expected an integer tuple, received {text!r}")
    return value


def flatten_gamma(parts: Sequence[tuple[int, ...]]) -> list[int]:
    return [value for part in parts for value in part]
