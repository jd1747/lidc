# src/preprocessing/nodule_size.py
# seg.nii.gz → 픽셀 개수로 부피 계산
# Small: < 100mm3
# python -m src.preprocessing.nodule_size
import os
import csv
import math
import logging
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# 기존 config에서 경로를 가져오거나 직접 지정 가능
from src.configs.config import PROCESSED_ROOT

NIFTI_SAVE_DIR = Path("/home/hce/Projects/lidc-idri/data/processed/nifti")
OUTPUT_CSV_PATH = PROCESSED_ROOT / "nodule_size_class.csv"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def classify_nodule_size(volume_mm3: float) -> str:
    """Fleischner Society 가이드라인 기반 보편적 크기 분류"""
    if volume_mm3 < 100:
        return "Small"
    elif volume_mm3 < 250:  # 기준 맞는지는 잘 모르겠음
        return "Medium"
    else:
        return "Large"


def process_all_segments():
    if not NIFTI_SAVE_DIR.exists():
        logger.error(f"NIfTI 저장 경로가 존재하지 않습니다: {NIFTI_SAVE_DIR}")
        return

    # CSV 파일 헤더 정의
    csv_headers = [
        "subject_id",
        "series_uid",
        "nodule_idx",
        "label_id",
        "voxel_count",
        "volume_mm3",
        "size_category",
    ]

    # 세그멘테이션 파일 검색 (seg.nii.gz)
    seg_files = list(NIFTI_SAVE_DIR.rglob("seg.nii.gz"))
    logger.info(f"총 {len(seg_files)}개의 세그멘테이션 파일을 찾았습니다. 분석을 시작합니다.")

    records = []

    for seg_path in tqdm(seg_files, desc="결절 통계 추출 중"):
        # 경로 파싱을 통해 subject_id와 series_uid 분리
        # 구조 1: NIFTI_SAVE_DIR / subject_id / series_uid / seg.nii.gz
        # 구조 2: NIFTI_SAVE_DIR / subject_id / seg.nii.gz
        relative_parts = seg_path.relative_to(NIFTI_SAVE_DIR).parts

        subject_id = relative_parts[0]
        series_uid = relative_parts[1] if len(relative_parts) == 3 else "single_series"

        try:
            # 1. 파일 로드
            seg_img = sitk.ReadImage(str(seg_path))
            seg_arr = sitk.GetArrayFromImage(seg_img)

            # 2. 복셀 간격 확인 (리샘플링으로 인해 1.0, 1.0, 1.0 이지만 방어적 계산)
            spacing = seg_img.GetSpacing()
            voxel_volume = spacing[0] * spacing[1] * spacing[2]

            # 3. 고유 라벨(결절) 추출 (0번 배경 제외)
            unique_labels = np.unique(seg_arr)
            unique_labels = unique_labels[unique_labels != 0]

            for label in unique_labels:
                # 각 라벨별 복셀 개수 연산
                voxel_count = int(np.sum(seg_arr == label))
                volume_mm3 = voxel_count * voxel_volume

                # 직경 및 크기 그룹 분류
                size_cat = classify_nodule_size(volume_mm3)

                records.append(
                    {
                        "subject_id": subject_id,
                        "series_uid": series_uid,
                        "nodule_idx": int(label - 1),  # 원본 json의 nodule_idx 복원
                        "label_id": int(label),
                        "voxel_count": voxel_count,
                        "volume_mm3": round(volume_mm3, 2),
                        "size_category": size_cat,
                    }
                )

        except Exception as e:
            logger.error(f"파일 처리 실패 ({seg_path}): {e}")

    # 4. CSV 쓰기
    with open(OUTPUT_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"추출 완료. 결과가 저장되었습니다: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    process_all_segments()
