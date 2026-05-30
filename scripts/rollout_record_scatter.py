"""Plot scatter of per-episode initial positions logged by rollout_record.py.

Reads `episodes.json` produced inside `rollout_videos/<checkpoint_name>/` and
draws a top-down (X/Y) scatter of the desk bounds, the robot arm position,
and the two cups' initial positions for every episode.

Usage:
    python scripts/rollout_record_scatter.py --json path/to/episodes.json
    python scripts/rollout_record_scatter.py --dir  rollout_videos/<checkpoint>
    python scripts/rollout_record_scatter.py --dir  rollout_videos/<checkpoint> --out scatter.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.json:
        json_path = Path(args.json).expanduser().resolve()
        if not json_path.is_file():
            raise SystemExit(f"episodes.json not found at {json_path}")
        out_path = Path(args.out).expanduser().resolve() if args.out else json_path.with_name("scatter.png")
        return json_path, out_path
    if args.dir:
        dir_path = Path(args.dir).expanduser().resolve()
        json_path = dir_path / "episodes.json"
        if not json_path.is_file():
            raise SystemExit(f"episodes.json not found under {dir_path}")
        out_path = Path(args.out).expanduser().resolve() if args.out else dir_path / "scatter.png"
        return json_path, out_path
    raise SystemExit("Provide --json or --dir")


def _xy(point):
    if not point or len(point) < 2:
        return None
    return float(point[0]), float(point[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None, help="Path to episodes.json.")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Path to a checkpoint folder under rollout_videos/.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output PNG path.")
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Label each cup point with its episode number.",
    )
    args = parser.parse_args()

    json_path, out_path = _resolve_paths(args)

    with json_path.open() as f:
        episodes = json.load(f)
    if not episodes:
        print(f"No episodes recorded in {json_path}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8, 8))

    desk = episodes[0].get("desk_range") or {}
    if desk:
        x0 = desk.get("x_min", 0.0)
        y0 = desk.get("y_min", 0.0)
        w = desk.get("x_max", 0.0) - x0
        h = desk.get("y_max", 0.0) - y0
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                w,
                h,
                linewidth=1.5,
                edgecolor="black",
                facecolor="lightgray",
                alpha=0.3,
                label="Desk",
            )
        )

    blue_xy, pink_xy, arm_xy = [], [], []
    blue_labels, pink_labels, arm_labels = [], [], []
    outcome_markers = {"success": "o", "timeout": "X", "manual_reset": "s", "incomplete": "D"}
    blue_markers, pink_markers = [], []

    for ep in episodes:
        ep_num = ep.get("episode", "?")
        outcome = ep.get("outcome", "")
        marker = outcome_markers.get(outcome, "o")

        bxy = _xy(ep.get("blue_cup_initial"))
        if bxy:
            blue_xy.append(bxy)
            blue_labels.append(f"{ep_num}")
            blue_markers.append(marker)
        pxy = _xy(ep.get("pink_cup_initial"))
        if pxy:
            pink_xy.append(pxy)
            pink_labels.append(f"{ep_num}")
            pink_markers.append(marker)
        axy = _xy(ep.get("robot_arm_position"))
        if axy:
            arm_xy.append(axy)
            arm_labels.append(f"{ep_num}")

    # Plot cups grouped by outcome so each marker style appears once in the legend.
    def _plot_group(points, markers, color, label_prefix):
        seen = set()
        for (x, y), m in zip(points, markers):
            label = None
            key = (color, m)
            if key not in seen:
                seen.add(key)
                outcome_name = next(
                    (name for name, mk in outcome_markers.items() if mk == m),
                    "?",
                )
                label = f"{label_prefix} ({outcome_name})"
            ax.scatter(
                x, y, c=color, s=70, marker=m, edgecolors="black", linewidths=0.6, label=label
            )

    _plot_group(blue_xy, blue_markers, "royalblue", "Blue cup")
    _plot_group(pink_xy, pink_markers, "deeppink", "Pink cup")
    if arm_xy:
        xs, ys = zip(*arm_xy)
        ax.scatter(
            xs,
            ys,
            c="forestgreen",
            s=90,
            marker="^",
            edgecolors="black",
            linewidths=0.6,
            label="Robot arm",
        )

    if args.annotate:
        for (x, y), lbl in zip(blue_xy, blue_labels):
            ax.annotate(lbl, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        for (x, y), lbl in zip(pink_xy, pink_labels):
            ax.annotate(lbl, (x, y), fontsize=7, xytext=(4, -8), textcoords="offset points")

    ax.set_xlabel("X (m, env-relative)")
    ax.set_ylabel("Y (m, env-relative)")
    ax.set_title(f"Episode initial positions — {json_path.parent.name}\n({len(episodes)} episodes)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved scatter to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
