# src/checks/check_patches_visual.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   labels_raw_{crop_size}.csv에서 샘플을 뽑아서
#   patch가 실제로 결절을 담고 있는지 시각화.
#
# ─── 사용 방법 ───────────────────────────────────────────────────────────────
#   python -m src.checks.check_patches --crop_size 64 --n_samples 16
#
# ─── 출력 ────────────────────────────────────────────────────────────────────
#   outputs/patch_check_{crop_size}.png

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.configs.config import PROCESSED_ROOT, OUTPUT_ROOT

NPY_CACHE_ROOT = PROCESSED_ROOT / "npy_cache"


def check_patches_visual(crop_size: int, n_samples: int, seed: int) -> None:
    """
    test fold에서 benign/malignant 각 절반씩 샘플링하여
    중심 슬라이스(채널 1)를 그리드로 시각화.
    """
    raw_csv = NPY_CACHE_ROOT / f"labels_raw_{crop_size}.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"CSV 없음: {raw_csv}")

    # test fold 행 로드
    rows = []
    with open(raw_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["fold"] == "test":
                rows.append(row)

    # benign / malignant 각 절반 샘플링
    random.seed(seed)
    benign = [r for r in rows if r["label"] == "0"]
    malignant = [r for r in rows if r["label"] == "1"]

    n_each = n_samples // 2
    sampled = random.sample(benign, min(n_each, len(benign))) + random.sample(malignant, min(n_each, len(malignant)))
    random.shuffle(sampled)

    # 그리드 크기 계산
    n_cols = 8
    n_rows = (len(sampled) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.2))
    axes = axes.flatten()

    for i, row in enumerate(sampled):
        patch = np.load(row["patch_path"])  # (3, H, W), float16
        center_slice = patch[1]  # 채널 1 = 중심 슬라이스 k

        ax = axes[i]
        ax.imshow(center_slice, cmap="gray", vmin=0, vmax=1)

        label_str = "MAL" if row["label"] == "1" else "BEN"
        diam = row.get("diameter_max_mm", "?")
        ax.set_title(f"{label_str} | d={diam}mm", fontsize=7, color="red" if row["label"] == "1" else "blue")
        ax.axis("off")

        # 중심점 표시 (patch 중심 = 결절 중심이어야 함)
        h, w = center_slice.shape
        ax.plot(w // 2, h // 2, "r+", markersize=8, markeredgewidth=1.5)

    # 남은 axes 숨기기
    for j in range(len(sampled), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Patch 시각화 | crop_size={crop_size} | 중심 슬라이스(k)\n"
        f"+ 표시 = patch 중심 (결절 중심이어야 함)\n"
        f"빨강=악성, 파랑=양성",
        fontsize=10,
    )
    fig.tight_layout()

    save_path = OUTPUT_ROOT / f"patch_check_{crop_size}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 저장: {save_path}")
    print(f"     결절이 + 표시 근처에 보여야 정상.")
    print(f"     + 주변이 비어있거나 폐 조직만 보이면 좌표 오류.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_patches_visual(args.crop_size, args.n_samples, args.seed)
