#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

IMG_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_ID_RE = re.compile(r"^(DJI_\d+_[A-Z])_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split YOLO dataset by source video id")
    p.add_argument("--zip", dest="zip_path", default="Tubes CSP.yolov11.zip")
    p.add_argument("--out", dest="out_dir", default="data/tubes_csp_yolov11_grouped")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--valid", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    out_dir = Path(args.out_dir)

    if abs((args.train + args.valid + args.test) - 1.0) > 1e-9:
        raise ValueError("train + valid + test must equal 1.0")
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split in ["train", "valid", "test"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        image_entries = [
            n for n in names if "/images/" in n and Path(n).suffix.lower() in IMG_EXTS
        ]

        label_set = {n for n in names if "/labels/" in n and n.endswith(".txt")}

        by_video: dict[str, list[tuple[str, str]]] = defaultdict(list)
        missing_labels = 0

        for img_path in image_entries:
            base = Path(img_path).stem
            label_path = str(Path("train") / "labels" / f"{base}.txt")
            if label_path not in label_set:
                missing_labels += 1
                continue

            m = VIDEO_ID_RE.match(base)
            video_id = m.group(1) if m else "unknown"
            by_video[video_id].append((img_path, label_path))

        # Greedy balance by total frame count to approximate ratios while keeping each video intact.
        rng = random.Random(args.seed)
        video_ids = list(by_video.keys())
        rng.shuffle(video_ids)
        video_ids.sort(key=lambda v: len(by_video[v]), reverse=True)

        total_frames = sum(len(v) for v in by_video.values())
        target = {
            "train": total_frames * args.train,
            "valid": total_frames * args.valid,
            "test": total_frames * args.test,
        }
        assigned = {"train": 0, "valid": 0, "test": 0}
        split_videos = {"train": [], "valid": [], "test": []}

        for vid in video_ids:
            n = len(by_video[vid])
            # Choose split that minimizes overflow vs target.
            best_split = min(
                ["train", "valid", "test"],
                key=lambda s: (assigned[s] + n - target[s], assigned[s] / (target[s] + 1e-9)),
            )
            assigned[best_split] += n
            split_videos[best_split].append(vid)

        # Materialize files
        split_counts = {"train": 0, "valid": 0, "test": 0}
        for split, vids in split_videos.items():
            for vid in vids:
                for img_path, label_path in by_video[vid]:
                    img_bytes = zf.read(img_path)
                    lbl_bytes = zf.read(label_path)

                    img_name = Path(img_path).name
                    lbl_name = Path(label_path).name

                    (out_dir / split / "images" / img_name).write_bytes(img_bytes)
                    (out_dir / split / "labels" / lbl_name).write_bytes(lbl_bytes)
                    split_counts[split] += 1

    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 2\n"
        "names: ['tube_ref', 'tube_test']\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("Split complete")
    print(f"zip: {zip_path}")
    print(f"output: {out_dir}")
    print(f"videos: {len(by_video)}")
    print(f"missing_labels_skipped: {missing_labels}")
    for s in ["train", "valid", "test"]:
        print(f"{s}: {split_counts[s]} images, {len(split_videos[s])} videos")
    print("data.yaml:")
    print((out_dir / "data.yaml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
