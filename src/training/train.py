# src/training/train.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   실험 설정 관리 + 학습 루프 실행.
#   실제 epoch 연산은 engine.py에 위임.
#
# ─── 사용 방법 ───────────────────────────────────────────────────────────────
#   conda run -n resnet --no-capture-output \
#     python -m src.training.train --model gdn --crop_size 32
#
#   conda run -n resnet --no-capture-output \
#     python -m src.training.train --model convnext --crop_size 64
#
#   conda run -n resnet --no-capture-output \
#     python -m src.training.train --model dual_convnext
#
# 1차 사용
# python -m src.training.train --model gdn --crop_size 32 --no_aug
#
# python -m src.training.train --model gdn --crop_size 32
#
# python -m src.training.train --model convnext --crop_size 64 --no_aug

# python -m src.training.train --model convnext --crop_size 64

# python -m src.training.train --model dual_convnext --no_aug

# python -m src.training.train --model dual_convnext
# nohup python -m src.training.train --model dual_convnext --augment hflip vflip rot90 > output_train.log 2>&1 &
#
# ─── 출력 구조 ───────────────────────────────────────────────────────────────
#   outputs/experiments/260608_gdn_32x32_ep50_aug1/
#     config.json     ← 실험 설정 전체 기록
#     history.csv     ← epoch별 train/val 지표
#     best_model.pth  ← val_auc 최고 시점
#     last_model.pth  ← 마지막 epoch

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

from src.configs.config import PROCESSED_ROOT, OUTPUT_ROOT, SEED
from src.datasets.dataset import get_dataloaders, get_dual_dataloaders

# from src.models.models_legacy import GDN, ConvNeXt, DualConvNeXt
from src.models.models import GDN, ConvNeXt, DualConvNeXt
from src.training.engine import train_one_epoch, validate_one_epoch
from src.utils.utils import get_device, set_seed

import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────────────────

NPY_CACHE_ROOT = PROCESSED_ROOT / "npy_cache"
EXP_ROOT = OUTPUT_ROOT / "experiments"  # config.py의 OUTPUT_ROOT 사용


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 1. 실험 폴더 생성
# ─────────────────────────────────────────────────────────────────────────────


# *수정 - crop_size 인자 int -> str로 수정
def make_exp_dir(model_name: str, crop_size: str, epochs: int, n_aug: int, title: str) -> Path:
    """
    실험 결과 저장 폴더 자동 생성.

    네이밍: 날짜_모델_crop_ep_aug수
    예: 260608_gdn_32x32_ep50_aug1
    OUTPUT_ROOT 기준으로 생성 → 실행 위치와 무관하게 항상 동일 경로.
    """
    date_str = datetime.now().strftime("%y%m%d")
    """
    결과
    260609_convnext_64_ep50_aug1
    260609_gdn_32_ep50_aug1
    260609_dual_convnext_32+96_ep50_aug1
    """
    if not title:
        exp_name = f"{date_str}_{model_name}_{crop_size}_ep{epochs}_aug{n_aug}"
    else:
        exp_name = f"{date_str}_{title}"
    exp_dir = EXP_ROOT / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def normalize_augmentations(args: argparse.Namespace) -> list[str]:
    """CLI augmentation 옵션을 학습에 사용할 리스트로 정규화."""
    if args.no_aug:
        return []
    return [a for a in args.augment if a != "none"]


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 2. 모델 빌드
# ─────────────────────────────────────────────────────────────────────────────


def build_model(model_name: str) -> nn.Module:
    """
    모델 이름으로 인스턴스 생성.

    모든 모델 출력: (B, 1) logit.
    BCEWithLogitsLoss 사용 → forward에 sigmoid 붙이지 않음.
    """
    if model_name == "gdn":
        return GDN(in_ch=3, num_classes=1)
    elif model_name == "convnext":
        return ConvNeXt(in_ch=3, num_classes=1)
    elif model_name == "dual_convnext":
        return DualConvNeXt(num_classes=1)
    elif model_name == "resnet":
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    else:
        raise ValueError(f"알 수 없는 모델: {model_name}\n선택 가능: gdn, convnext, dual_convnext")


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 3. pos_weight 자동 계산
# *수정
"""
 pos_weight = compute_pos_weight(args.crop_size) 이고
 python train.py --model dual_convnext 실행하면
 args.crop_size == 64 결국 기본값 사용해서
 labers_aug_64.csv를 찾게 됨 파일 없을 시 2.0으로 fallback하는 문제
"""
# ─────────────────────────────────────────────────────────────────────────────

# def compute_pos_weight(crop_size: int) -> torch.Tensor:
#     """
#     labels_aug_{crop_size}.csv에서 train fold 클래스 비율로 pos_weight 계산.

#     공식: pos_weight = n_benign / n_malignant
#     효과: malignant 샘플 loss 기여도를 높여서 2:1 불균형 보정.

#     왜 자동 계산인가:
#       데이터 변경(증강 추가, fold 재분할) 시 자동 반영.
#       고정값이면 변경 때마다 직접 수정해야 하는 실수 위험.
#     """
#     csv_path = NPY_CACHE_ROOT / f'labels_aug_{crop_size}.csv'
#     if not csv_path.exists():
#         print(f'[WARN] labels_aug_{crop_size}.csv 없음. pos_weight=2.0 기본값 사용.')
#         return torch.tensor([2.0])

#     n_benign, n_malignant = 0, 0
#     with open(csv_path, 'r', encoding='utf-8') as f:
#         for row in csv.DictReader(f):
#             if row['fold'] == 'train':
#                 if row['label'] == '0':
#                     n_benign += 1
#                 elif row['label'] == '1':
#                     n_malignant += 1

#     if n_malignant == 0:
#         print('[WARN] malignant 샘플 없음. pos_weight=2.0 기본값 사용.')
#         return torch.tensor([2.0])

#     pw = n_benign / n_malignant
#     print(f'[INFO] pos_weight 자동 계산: {n_benign}/{n_malignant} = {pw:.4f}')
#     return torch.tensor([pw])


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 4. 결과 저장 유틸리티
# ─────────────────────────────────────────────────────────────────────────────


# *수정
def save_config(
    exp_dir: Path, args: argparse.Namespace, pos_weight: float, crop_info, augmentations: list[str]
) -> None:
    """실험 설정 전체를 config.json으로 저장."""
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "crop_size": crop_info,  # *수정
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight": round(pos_weight, 4),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss": "BCEWithLogitsLoss",
        "augmentations": augmentations,
        "aug_prob": args.aug_prob,
        "n_slices": 1,
        "stride": 1,
        "seed": args.seed,
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[INFO] config.json 저장: {exp_dir / 'config.json'}")


class HistoryWriter:
    """
    history.csv에 epoch별 지표를 한 줄씩 기록.

    매 epoch 즉시 기록 → 학습 중단 시에도 기록 보존.

    컬럼:
      epoch,
      train_loss/auc/accuracy/sensitivity/specificity,
      val_loss/auc/accuracy/sensitivity/specificity,
      best_auc_flag
    """

    FIELDNAMES = [
        "epoch",
        "train_loss",
        "train_auc",
        "train_accuracy",
        "train_sensitivity",
        "train_specificity",
        "val_loss",
        "val_auc",
        "val_accuracy",
        "val_sensitivity",
        "val_specificity",
        "best_auc_flag",
        "lr",  # *수정 lr 추가
    ]

    def __init__(self, exp_dir: Path):
        self.csv_path = exp_dir / "history.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def write(self, epoch: int, train_metrics: dict, val_metrics: dict, is_best: bool, lr: float) -> None:
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "train_accuracy": train_metrics["accuracy"],
            "train_sensitivity": train_metrics["sensitivity"],
            "train_specificity": train_metrics["specificity"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "best_auc_flag": 1 if is_best else 0,
            "lr": lr,  # *수정 lr 추가
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)


# *수정 - alpha값 호출부
class AlphaWriter:
    LAYER_NAMES = ["gd1", "gd2", "gd3", "gd4", "gd5"]

    def __init__(self, exp_dir: Path):
        self.csv_path = exp_dir / "alpha_history.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["epoch"] + self.LAYER_NAMES).writeheader()

    def write(self, epoch: int, alpha_means: dict) -> None:
        row = {"epoch": epoch}
        row.update({k: round(alpha_means.get(k, 0.0), 4) for k in self.LAYER_NAMES})
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["epoch"] + self.LAYER_NAMES).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 5. 메인 학습 루프
# ─────────────────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    """
    전체 학습 파이프라인 실행.

    순서:
      1. seed 고정 + device 설정
      2. 실험 폴더 생성
      3. DataLoader 빌드
      4. 모델 / optimizer / scheduler / loss 설정
      5. epoch 루프
      6. best_model.pth / last_model.pth 저장
    """
    set_seed(args.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = get_device()
    is_dual = args.model == "dual_convnext"

    augmentations = normalize_augmentations(args)
    n_aug = len(augmentations)
    # display_crop = "32+96" if is_dual else str(args.crop_size)
    display_crop = "48+96" if is_dual else str(args.crop_size)
    exp_dir = make_exp_dir(args.model, display_crop, args.epochs, n_aug, args.title)
    print(f"[INFO] 실험 폴더: {exp_dir}")

    # DataLoader
    if is_dual:
        train_loader, val_loader, _, pos_weight = get_dual_dataloaders(
            # crop_size_small=32,
            crop_size_small=48,
            crop_size_large=96,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augmentations=augmentations,
            aug_prob=args.aug_prob,
        )
    else:
        train_loader, val_loader, _, pos_weight = get_dataloaders(
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augmentations=augmentations,
            aug_prob=args.aug_prob,
        )

    # 모델
    model = build_model(args.model).to(device)
    print(f"[INFO] 모델: {args.model} | 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # pos_weight = pos_weight.to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    pos_weight = torch.tensor([1.0], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer: AdamW (weight decay를 gradient와 분리 → 정확한 regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler: CosineAnnealingLR (lr을 코사인 곡선으로 부드럽게 감소)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_config(exp_dir, args, float(pos_weight.item()), display_crop, augmentations)
    history_writer = HistoryWriter(exp_dir)
    alpha_writer = AlphaWriter(exp_dir) if args.model == "gdn" else None  # 추가

    best_val_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, is_dual=is_dual)
        val_metrics, alpha_means = validate_one_epoch(model, val_loader, criterion, device, is_dual=is_dual)

        if alpha_writer is not None and alpha_means:
            alpha_writer.write(epoch, alpha_means)
            print("[ALPHA] " + " ".join(f"{k}={v:.3f}" for k, v in alpha_means.items()))

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_metrics["auc"] > best_val_auc
        if is_best:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), exp_dir / "best_model.pth")

        # *수정
        history_writer.write(epoch, train_metrics, val_metrics, is_best, lr=current_lr)

        print(
            f"[Epoch {epoch:3d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_auc={train_metrics['auc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f}"
            f"{' ← best' if is_best else ''}"
        )

    torch.save(model.state_dict(), exp_dir / "last_model.pth")
    print(f"\n[DONE] 학습 완료 | best val_auc: {best_val_auc:.4f}")
    print(f"평가 실행: python -m src.evaluation.evaluate --exp_dir {exp_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 6. 명령줄 인터페이스
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIDC-IDRI 결절 분류 학습")
    parser.add_argument("--model", type=str, default="gdn", choices=["gdn", "convnext", "dual_convnext", "resnet"])
    parser.add_argument(
        "--no_aug", action="store_true", help="증강 없이 raw 데이터만 사용 (baseline용, --augment none과 동일)"
    )
    parser.add_argument(
        "--augment",
        type=str,
        nargs="+",
        default=["none"],
        choices=["none", "hflip", "vflip", "rot90", "hu_shift", "gaussian_noise"],
        help="train fold에 online augmentation 적용. 예: --augment hflip 또는 --augment hflip gaussian_noise",
    )
    parser.add_argument("--aug_prob", type=float, default=0.5, help="선택한 augmentation별 적용 확률")
    parser.add_argument(
        "--crop_size", type=int, default=32, help="gdn=32, convnext=64 권장. dual_convnext는 48+96 자동 사용."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4, help="GPU 사용 시 4~8 권장")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--title", type=str, default="")
    return parser.parse_args()


# python -m src.training.train --no_aug --title gdn32_aug0
# nohup python -m src.training.train --augment hflip rot90 --title gdn32_geo > gdn32_geo.log 2>&1 &
# python -m src.training.train --augment hflip rot90 hu_shift gaussian_noise --title gdn32_all

if __name__ == "__main__":
    args = parse_args()
    print(
        f"[CONFIG] model={args.model} | crop_size={args.crop_size} | "
        f"epochs={args.epochs} | batch_size={args.batch_size} | lr={args.lr} | "
        f"augment={normalize_augmentations(args)}"
    )
    # train(args)

    seeds = [1, 3, 13, 22, 42]
    for i in range(5):
        args.seed = seeds[i]
        args.title = f"{args.title}_seed_{seeds[i]}"
        train(args)
