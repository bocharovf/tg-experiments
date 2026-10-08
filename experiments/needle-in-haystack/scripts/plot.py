"""Plot needle-in-a-haystack results: accuracy vs context position.

Reads the result CSVs (rows must have a `task` column) and draws, for each model
and task, the recall curve by position, one line per prompt style. This is the
classic "lost in the middle" chart.

Usage:
    python scripts/plot.py                     # results/*.csv
    python scripts/plot.py --paths a.csv b.csv --out results/plot.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE_COLORS = {"neutral": "#1f77b4", "primed": "#d62728"}
STYLE_MARKERS = {"neutral": "o", "primed": "s"}

# Snap measured positions (which drift slightly with document length) onto the
# fixed 12-marker grid so models with different haystack sizes align on one axis.
N_GRID = 12


def snap_pos(p: int) -> int:
    return round(round(p * (N_GRID - 1) / 100) * 100 / (N_GRID - 1))


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with p.open(encoding="utf-8", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return [r for r in rows if r.get("task")]


def accuracy(rows: list[dict]) -> dict[tuple[str, str, str, int], float]:
    """model, task, prompt_style, pos_a -> mean correct."""
    agg: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["task"], r["prompt_style"], snap_pos(int(r["pos_a"])))
        agg[key].append(int(r["correct"]))
    return {k: 100 * sum(v) / len(v) for k, v in agg.items()}


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", nargs="+", type=Path, help="CSV files (default: results/*.csv)")
    parser.add_argument("--out", type=Path, default=Path("results/plot.png"))
    args = parser.parse_args()

    paths = args.paths or sorted(Path("results").glob("*.csv"))
    if not paths:
        raise SystemExit("no CSV files found")
    rows = load(paths)
    if not rows:
        raise SystemExit("no rows with a 'task' column")
    acc = accuracy(rows)

    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    styles = sorted({r["prompt_style"] for r in rows})

    fig, axes = plt.subplots(len(models), len(tasks), figsize=(6.5 * len(tasks), 4.2 * len(models)),
                             squeeze=False, sharex=True, sharey=True)
    for mi, model in enumerate(models):
        for ti, task in enumerate(tasks):
            ax = axes[mi][ti]
            positions = sorted({p for (m, t, s, p) in acc if m == model and t == task})
            for style in styles:
                ys = [acc.get((model, task, style, p)) for p in positions]
                ax.plot(positions, ys, marker=STYLE_MARKERS.get(style, "o"),
                        color=STYLE_COLORS.get(style), label=style, linewidth=2, markersize=6)
            ax.set_title(f"{model} — task={task}")
            ax.set_xlabel("position of marker in context (%)")
            ax.set_ylabel("accuracy (%)")
            ax.set_ylim(-2, 102)
            ax.set_xlim(-2, 102)
            ax.grid(True, alpha=0.3)
            if mi == 0 and ti == 0:
                ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=len(styles))
            for p, y in zip(positions, ys):
                ax.annotate(f"{y:.0f}", (p, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    fig.suptitle("Needle-in-a-haystack — does recall drop in the middle of a ~115K-token context?",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=130)
    print(f"saved {args.out}")

    # also print a text table
    print(f"\naccuracy (%) by position — {', '.join(models)}")
    for task in tasks:
        print(f"\n== task={task}")
        positions = sorted({p for (_m, t, _s, p) in acc if t == task})
        print("pos | " + " | ".join(f"{m[:22]:<22}" for m in models))
        for p in positions:
            cells = []
            for m in models:
                parts = [f"{s[0]}:{acc.get((m, task, s, p), 0):.0f}" for s in styles]
                cells.append(" ".join(parts))
            print(f"{p:>3} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
