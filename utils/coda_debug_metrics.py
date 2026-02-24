from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_json_rows(pattern: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                row = json.load(f)
            if isinstance(row, dict):
                rows.append(row)
        except Exception:
            continue
    return rows


def _arr(rows: List[Dict[str, Any]], key: str, default: float = np.nan) -> np.ndarray:
    out: List[float] = []
    for row in rows:
        value = row.get(key, default)
        try:
            out.append(float(value))
        except Exception:
            out.append(float(default))
    return np.asarray(out, dtype=np.float64)


def _dict_arr(rows: List[Dict[str, Any]], dict_key: str, sub_key: str, default: float = 0.0) -> np.ndarray:
    out: List[float] = []
    for row in rows:
        raw = row.get(dict_key, {})
        if not isinstance(raw, dict):
            out.append(float(default))
            continue
        value = raw.get(sub_key, default)
        try:
            out.append(float(value))
        except Exception:
            out.append(float(default))
    return np.asarray(out, dtype=np.float64)


def _plot_metrics(rows: List[Dict[str, Any]], title: str):
    steps = _arr(rows, "step")
    train_loss = _arr(rows, "train_loss")
    pred_err = _arr(rows, "predicted_reconstruction_error")
    recon_err = _arr(rows, "reconstruction_error")
    active_slots = _arr(rows, "active_slots", default=0.0)
    gate_mean = _arr(rows, "gate_mean")
    updates = _arr(rows, "total_updates", default=0.0)

    active_agent = _dict_arr(rows, "active_per_type", "agent", default=0.0)
    active_pod = _dict_arr(rows, "active_per_type", "pod", default=0.0)
    active_wall = _dict_arr(rows, "active_per_type", "wall", default=0.0)
    active_empty = _dict_arr(rows, "active_per_type", "empty", default=0.0)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(steps, train_loss, marker="o", linewidth=1.5, label="train_loss")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(steps, pred_err, marker="o", linewidth=1.5, label="predicted_reconstruction_error")
    ax.plot(steps, recon_err, marker="o", linewidth=1.5, label="reconstruction_error")
    ax.set_ylabel("Error")
    ax.set_title("Reconstruction Errors")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(steps, active_slots, marker="o", linewidth=1.5, label="active_slots")
    ax.set_ylabel("Count")
    ax.set_title("Active Slot Count")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.stackplot(
        steps,
        active_agent,
        active_pod,
        active_wall,
        active_empty,
        labels=["agent", "pod", "wall", "empty"],
        alpha=0.8,
    )
    ax.set_ylabel("Count")
    ax.set_title("Active Slots by Type")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left")

    ax = axes[2, 0]
    ax.plot(steps, gate_mean, marker="o", linewidth=1.5, label="gate_mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gate")
    ax.set_title("Mean Gate")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    ax = axes[2, 1]
    ax.plot(steps, updates, marker="o", linewidth=1.5, label="total_updates")
    ax.set_xlabel("Step")
    ax.set_ylabel("Updates")
    ax.set_title("Optimizer Updates")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot CODA debug metrics from JSON dumps.")
    parser.add_argument(
        "--pattern",
        default="data/coda_debug/*.json",
        help="Glob pattern for CODA metric JSON files.",
    )
    parser.add_argument(
        "--title",
        default="CODA Debug Metrics",
        help="Plot title.",
    )
    args = parser.parse_args()

    rows = _load_json_rows(args.pattern)
    if not rows:
        print(f"No metric files found for pattern: {args.pattern}")
        return

    print(f"Loaded {len(rows)} metric files from: {os.path.dirname(args.pattern) or '.'}")
    _plot_metrics(rows, title=args.title)


if __name__ == "__main__":
    main()
