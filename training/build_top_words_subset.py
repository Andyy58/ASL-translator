#!/usr/bin/env python3
"""
Build a stronger WLASL feature subset from already-extracted .npy files.

The script pools train/val/test files for each gloss, selects the most populated
glosses, and writes a fresh train/val split. The original feature folders are
left untouched.
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class WordStats:
    word: str
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test

    @property
    def train_plus_test(self) -> int:
        return self.train + self.test

    @property
    def has_validation(self) -> int:
        return int(self.val > 0)


def list_files(input_root: Path, word: str) -> list[Path]:
    files: list[Path] = []
    for split in SPLITS:
        split_dir = input_root / split / word
        if split_dir.exists():
            files.extend(sorted(split_dir.glob("*.npy")))
    return files


def collect_stats(input_root: Path) -> list[WordStats]:
    words: set[str] = set()
    for split in SPLITS:
        split_dir = input_root / split
        if split_dir.exists():
            words.update(path.name for path in split_dir.iterdir() if path.is_dir())

    stats: list[WordStats] = []
    for word in sorted(words):
        counts = {}
        for split in SPLITS:
            word_dir = input_root / split / word
            counts[split] = len(list(word_dir.glob("*.npy"))) if word_dir.exists() else 0
        stats.append(
            WordStats(
                word=word,
                train=counts["train"],
                val=counts["val"],
                test=counts["test"],
            )
        )
    return stats


def select_words(stats: list[WordStats], num_words: int, min_total: int) -> list[WordStats]:
    eligible = [item for item in stats if item.total >= min_total]

    # Prefer words that were represented in both training and validation, then
    # prefer total pooled examples. This keeps the subset populated and balanced.
    ranked = sorted(
        eligible,
        key=lambda item: (
            item.has_validation,
            min(item.train_plus_test, item.val),
            item.total,
            item.train_plus_test,
            item.word,
        ),
        reverse=True,
    )
    return ranked[:num_words]


def clean_output(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / split).mkdir(parents=True, exist_ok=True)


def split_files(files: list[Path], val_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    shuffled = files[:]
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, []

    val_count = max(1, round(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:]
    return train_files, val_files


def copy_or_link(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def build_subset(
    input_root: Path,
    output_root: Path,
    selected: list[WordStats],
    val_ratio: float,
    seed: int,
    symlink: bool,
) -> None:
    rng = random.Random(seed)
    clean_output(output_root)

    for item in selected:
        files = list_files(input_root, item.word)
        train_files, val_files = split_files(files, val_ratio, rng)

        for split, split_files_ in (("train", train_files), ("val", val_files)):
            for src in split_files_:
                dst = output_root / split / item.word / src.name
                copy_or_link(src, dst, symlink=symlink)


def write_manifest(output_root: Path, selected: list[WordStats], val_ratio: float, seed: int) -> None:
    manifest_path = output_root / "selected_words.tsv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("rank\tword\tpooled_total\toriginal_train\toriginal_val\toriginal_test\n")
        for rank, item in enumerate(selected, start=1):
            handle.write(
                f"{rank}\t{item.word}\t{item.total}\t{item.train}\t{item.val}\t{item.test}\n"
            )

    readme_path = output_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# WLASL Top-Words Feature Subset",
                "",
                "Generated from pooled `train`, `val`, and `test` feature files.",
                f"Validation ratio: `{val_ratio}`",
                f"Shuffle seed: `{seed}`",
                "",
                "Use this folder with `ASLFeatureDataset`, for example:",
                "",
                "```python",
                'train_dataset = ASLFeatureDataset("../wlasl1000_top100_features/train", max_frames=100)',
                'validation_dataset = ASLFeatureDataset("../wlasl1000_top100_features/val", max_frames=100)',
                "```",
                "",
                "Selected words are listed in `selected_words.tsv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("wlasl1000_features"))
    parser.add_argument("--output-root", type=Path, default=Path("wlasl1000_top100_features"))
    parser.add_argument("--num-words", type=int, default=100)
    parser.add_argument("--min-total", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying .npy files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_root.exists():
        raise SystemExit(f"Input root does not exist: {args.input_root}")
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1")

    stats = collect_stats(args.input_root)
    selected = select_words(stats, num_words=args.num_words, min_total=args.min_total)
    if len(selected) < args.num_words:
        raise SystemExit(
            f"Only found {len(selected)} eligible words with at least {args.min_total} files."
        )

    build_subset(
        input_root=args.input_root,
        output_root=args.output_root,
        selected=selected,
        val_ratio=args.val_ratio,
        seed=args.seed,
        symlink=args.symlink,
    )
    write_manifest(args.output_root, selected, val_ratio=args.val_ratio, seed=args.seed)

    train_count = len(list((args.output_root / "train").glob("*/*.npy")))
    val_count = len(list((args.output_root / "val").glob("*/*.npy")))
    print(f"Wrote {args.output_root}")
    print(f"Selected words: {len(selected)}")
    print(f"Train files: {train_count}")
    print(f"Validation files: {val_count}")
    print(f"Manifest: {args.output_root / 'selected_words.tsv'}")


if __name__ == "__main__":
    main()
