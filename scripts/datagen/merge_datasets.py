"""Merge multiple LeRobot datasets into a single unified dataset.

Thin CLI wrapper around ``lerobot.datasets.aggregate.aggregate_datasets`` that
adds friendly pre-/post-merge reporting:

  * Before merging, prints each source dataset's fps / robot_type / episode &
    frame counts / feature keys, and fails early (with a readable diff) if the
    datasets are not mergeable.
  * After merging, reloads the aggregated dataset and checks that its episode
    and frame totals equal the sum of the sources.

Datasets are resolved from the default LeRobot cache
(``~/.cache/huggingface/lerobot/<repo_id>``) unless ``--roots`` is given.
The aggregated dataset is written to a brand-new ``--aggr_repo_id``; the source
datasets are never modified.

Usage:
python scripts/datagen/merge_datasets.py \
    --repo_ids [dataset repo 1] [dataset repo 2] \
    --aggr_repo_id [new dataset repo]

Eg. 
python scripts/datagen/merge_datasets.py \
    --repo_ids YinXuanLi/gen_dataset qiaoceng/AIC-data_augment-v2 \
    --aggr_repo_id qiaoceng/AIC-data_merged
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets into one.")
    parser.add_argument(
        "--repo_ids",
        type=str,
        nargs="+",
        required=True,
        help="Two or more source dataset repo_ids to merge (order is preserved).",
    )
    parser.add_argument(
        "--aggr_repo_id",
        type=str,
        required=True,
        help="repo_id for the merged output dataset. Must be a NEW id (not an existing source).",
    )
    parser.add_argument(
        "--roots",
        type=str,
        nargs="+",
        default=None,
        help="Optional local root paths for each source, in the same order as --repo_ids. "
        "Omit to use the default LeRobot cache (~/.cache/huggingface/lerobot/<repo_id>).",
    )
    parser.add_argument(
        "--aggr_root",
        type=str,
        default=None,
        help="Optional local root path for the merged dataset (default: LeRobot cache).",
    )
    return parser.parse_args()


def _load_sources(repo_ids: list[str], roots: list[str] | None) -> list[LeRobotDatasetMetadata]:
    if roots is not None and len(roots) != len(repo_ids):
        sys.exit(f"--roots has {len(roots)} entries but --repo_ids has {len(repo_ids)}; they must match.")

    metas: list[LeRobotDatasetMetadata] = []
    for idx, repo_id in enumerate(repo_ids):
        root = Path(roots[idx]) if roots is not None else None
        try:
            metas.append(LeRobotDatasetMetadata(repo_id, root=root))
        except Exception as e:  # noqa: BLE001 - surface a readable message instead of a traceback
            sys.exit(f"Failed to load source dataset '{repo_id}' (root={root}): {e}")
    return metas


def _report_sources(metas: list[LeRobotDatasetMetadata]) -> tuple[int, int]:
    """Print a per-source summary and return (sum_episodes, sum_frames)."""
    total_episodes = 0
    total_frames = 0
    print("=" * 70)
    print(f"Merging {len(metas)} source datasets:")
    for m in metas:
        print(f"\n  [{m.repo_id}]")
        print(f"    root          : {m.root}")
        print(f"    robot_type    : {m.robot_type}")
        print(f"    fps           : {m.fps}")
        print(f"    total_episodes: {m.total_episodes}")
        print(f"    total_frames  : {m.total_frames}")
        print(f"    feature keys  : {sorted(m.features.keys())}")
        total_episodes += m.total_episodes
        total_frames += m.total_frames
    print("=" * 70)
    return total_episodes, total_frames


def _check_mergeable(metas: list[LeRobotDatasetMetadata]) -> None:
    """Fail early with a readable diff if datasets are not mergeable.

    ``aggregate_datasets`` itself validates this, but its ValueError does not
    say *what* differs; this gives a clearer message before any work happens.
    """
    ref = metas[0]
    problems: list[str] = []
    for m in metas[1:]:
        if m.fps != ref.fps:
            problems.append(f"fps mismatch: '{ref.repo_id}'={ref.fps} vs '{m.repo_id}'={m.fps}")
        if m.robot_type != ref.robot_type:
            problems.append(
                f"robot_type mismatch: '{ref.repo_id}'={ref.robot_type} vs '{m.repo_id}'={m.robot_type}"
            )
        ref_keys, m_keys = set(ref.features), set(m.features)
        if ref_keys != m_keys:
            only_ref = sorted(ref_keys - m_keys)
            only_m = sorted(m_keys - ref_keys)
            problems.append(
                f"feature keys differ between '{ref.repo_id}' and '{m.repo_id}': "
                f"only in first={only_ref}, only in second={only_m}"
            )
        else:
            for key in ref_keys:
                ref_dtype = ref.features[key].get("dtype")
                m_dtype = m.features[key].get("dtype")
                ref_shape = ref.features[key].get("shape")
                m_shape = m.features[key].get("shape")
                if ref_dtype != m_dtype or ref_shape != m_shape:
                    problems.append(
                        f"feature '{key}' differs between '{ref.repo_id}' and '{m.repo_id}': "
                        f"{ref_dtype}{ref_shape} vs {m_dtype}{m_shape}"
                    )

    if problems:
        print("\n[ERROR] Datasets are NOT mergeable:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("[OK] fps, robot_type and features are consistent across all sources.\n")


def _verify_result(aggr_repo_id: str, aggr_root: str | None, expected_episodes: int, expected_frames: int) -> None:
    root = Path(aggr_root) if aggr_root is not None else None
    merged = LeRobotDatasetMetadata(aggr_repo_id, root=root)
    print("\n" + "=" * 70)
    print(f"Merged dataset: [{merged.repo_id}]")
    print(f"  root          : {merged.root}")
    print(f"  total_episodes: {merged.total_episodes} (expected {expected_episodes})")
    print(f"  total_frames  : {merged.total_frames} (expected {expected_frames})")
    print("=" * 70)

    ok = merged.total_episodes == expected_episodes and merged.total_frames == expected_frames
    if ok:
        print("[OK] Merge verified: episode and frame totals match the sum of sources.")
    else:
        print("[WARN] Totals do NOT match the sum of sources; please inspect the merged dataset.")
        sys.exit(1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    if len(args.repo_ids) < 2:
        sys.exit("Need at least two --repo_ids to merge.")
    if args.aggr_repo_id in args.repo_ids:
        sys.exit(f"--aggr_repo_id '{args.aggr_repo_id}' must be a NEW id, not one of the sources.")

    metas = _load_sources(args.repo_ids, args.roots)
    expected_episodes, expected_frames = _report_sources(metas)
    _check_mergeable(metas)

    aggregate_datasets(
        repo_ids=args.repo_ids,
        aggr_repo_id=args.aggr_repo_id,
        roots=[Path(r) for r in args.roots] if args.roots is not None else None,
        aggr_root=Path(args.aggr_root) if args.aggr_root is not None else None,
    )

    _verify_result(args.aggr_repo_id, args.aggr_root, expected_episodes, expected_frames)


if __name__ == "__main__":
    main()
