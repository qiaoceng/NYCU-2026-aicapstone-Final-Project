#!/usr/bin/env python3
"""Parse a LeRobot training output.log and plot the loss (and grad norm) curves.

Usage:
    python plot_loss.py [path/to/output.log] [-o out.png]

The log lines look like:
    Training: 99%|...| 98800/100000 [...]INFO ... step:99K ... loss:0.044 grdn:1.371 lr:1.0e-05 ...
We take the exact step from the progress bar ("98800/100000"), since the
"step:99K" field is rounded and unsuitable for plotting.
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# exact step from the tqdm progress bar, e.g. "| 98800/100000 ["
STEP_RE = re.compile(r"\|\s*(\d+)/\d+\s*\[")
LOSS_RE = re.compile(r"loss:([\d.eE+-]+)")
GRDN_RE = re.compile(r"grdn:([\d.eE+-]+)")


def parse_log(path: Path):
    steps, losses, grdns = [], [], []
    for line in path.read_text().splitlines():
        m_loss = LOSS_RE.search(line)
        m_step = STEP_RE.search(line)
        if not (m_loss and m_step):
            continue
        steps.append(int(m_step.group(1)))
        losses.append(float(m_loss.group(1)))
        m_grdn = GRDN_RE.search(line)
        grdns.append(float(m_grdn.group(1)) if m_grdn else float("nan"))
    return steps, losses, grdns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "log",
        nargs="?",
        default="/mnt/HDD5/college_student/aicapstone-Final-Project/checkpoints/output.log",
        help="path to output.log",
    )
    ap.add_argument("-o", "--out", default=None, help="output image path")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.with_name("loss_curve.png")

    steps, losses, grdns = parse_log(log_path)
    if not steps:
        raise SystemExit(f"No loss data found in {log_path}")

    print(f"Parsed {len(steps)} points  (step {steps[0]} -> {steps[-1]})")
    print(f"loss: first={losses[0]:.4f}  last={losses[-1]:.4f}  min={min(losses):.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(steps, losses, lw=1.5, label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.tick_params(axis="y")
    ax1.grid(True, alpha=0.3)

    plt.title(f"Training loss — {log_path.parent.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    main()
