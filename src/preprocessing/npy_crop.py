# src/preprocessing/npy_crop.py
#
# ─── 역할 ─────────────────────────────────────────────────────────
# - 결절 들어간 슬라이스 crop (고정 크기)
# -
# -
# -
#
# ─── 파이프라인 상 위치 ─────────────────────────────────────────────────
# [1] match_raw.py
# [2] nodule_segment.py
# [3] [현재] npy_crop.py
#
# ─── 출력 ─────────────────────────────────────────────────────────────
#
#
# ─── 실행 ─────────────────────────────────────────────────────────────
# python -m src.preprocessing.npy_crop
# nohup python -m src.preprocessing.npy_crop > output_crop.log 2>&1 &
import os
import json
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

from src.configs.config import NIFTI_SAVE_DIR, PATCHES_ROOT, JSON_PATH

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


CROP_SIZE = 96

MIN_HU = -1000  # 공기 HU (하한)
MAX_HU = 400  # 연조직/뼈 경계 (상한)


# ──────────────────────────────────────────────────────────────────────
# 1. HU 윈도잉 및 [0, 1] 정규화 필터
# ──────────────────────────────────────────────────────────────────────
def apply_lung_window(volume_slice: np.ndarray) -> np.ndarray:
    clipped = np.clip(volume_slice, MIN_HU, MAX_HU)
    normalized = (clipped - MIN_HU) / (MAX_HU - MIN_HU)
    return normalized.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# 2. 2D 고정 크기 Crop 유틸리티
# ──────────────────────────────────────────────────────────────────────
def safe_crop_2d(
    slice_2d: np.ndarray, center_y: int, center_x: int, crop_size: int = 96, pad_value: float = 0.0
) -> np.ndarray:
    """
    지정된 (center_y, center_x)를 중심축으로 고정 크기 Crop 수행.
    결절이 가쪽 경계면에 걸려 인덱스를 벗어나더라도 pad_value로 안전하게 여백(Padding)을 채움.
    """
    half = crop_size // 2
    y_min, y_max = center_y - half, center_y + half
    x_min, x_max = center_x - half, center_x + half

    # 이미지 경계 초과분 연산
    pad_y_pre = max(0, -y_min)
    pad_y_post = max(0, y_max - slice_2d.shape[0])
    pad_x_pre = max(0, -x_min)
    pad_x_post = max(0, x_max - slice_2d.shape[1])

    # 안전 구역 내부만 1차 슬라이싱
    cropped = slice_2d[max(0, y_min) : min(slice_2d.shape[0], y_max), max(0, x_min) : min(slice_2d.shape[1], x_max)]

    # 결손 부위 패딩 적용
    if pad_y_pre > 0 or pad_y_post > 0 or pad_x_pre > 0 or pad_x_post > 0:
        cropped = np.pad(
            cropped, ((pad_y_pre, pad_y_post), (pad_x_pre, pad_x_post)), mode="constant", constant_values=pad_value
        )
    return cropped


# ──────────────────────────────────────────────────────────────────────
# 3. 메인 프로세스
# ──────────────────────────────────────────────────────────────────────
def build_patch_dataset():
    """전체 환자 폴더 순회 → 2.5D/3D 축차 투입용 단면 슬라이스 전수 추출 및 마스터 매니페스트 백업"""
    if not JSON_PATH.exists():
        logger.error(f"기반 정보 파일이 없습니다: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        patient_dict = json.load(f)

    manifest_records = []
    logger.info(f"패치 데이터셋 빌드 시작 -> 대상 환자: {len(patient_dict)}명")

    for subject_id, patient_info in tqdm(patient_dict.items(), desc="NPY 패치 추출 중"):
        subj_nifti_dir = NIFTI_SAVE_DIR / subject_id
        ct_path = subj_nifti_dir / "ct.nii.gz"
        seg_path = subj_nifti_dir / "seg.nii.gz"

        if not ct_path.exists() or not seg_path.exists():
            continue

        # 3D 볼륨 로드 및 데이터 레이아웃 정합 (Z, Y, X 순 정렬)
        ct_img = sitk.ReadImage(str(ct_path))
        seg_img = sitk.ReadImage(str(seg_path))

        ct_arr = sitk.GetArrayFromImage(ct_img)
        seg_arr = sitk.GetArrayFromImage(seg_img)

        # 환자 전용 출력 디렉토리 확보
        subj_patch_dir = PATCHES_ROOT / subject_id
        subj_patch_dir.mkdir(parents=True, exist_ok=True)

        for nodule in patient_info.get("nodules", []):
            nodule_idx = nodule["nodule_idx"]
            malignancy_list = nodule["malignancy"]

            flat_malignancy = [score for sublist in malignancy_list for score in sublist]
            avg_malignancy = round(np.median(flat_malignancy), 2) if flat_malignancy else 0.0

            # segment.py 마스크 빌드 규칙과 일치하는 라벨 ID 추적
            label_id = nodule_idx + 1

            # 결절의 인덱스 좌표 위치 전수 수집
            z_indices, y_indices, x_indices = np.where(seg_arr == label_id)
            if len(z_indices) == 0:
                continue

            # 결절이 속한 모든 슬라이스를 대변하는 '고정 중심점' 계산
            crop_center_y = int(round(np.mean(y_indices)))
            crop_center_x = int(round(np.mean(x_indices)))

            # 결절이 포함된 슬라이스들 확보
            unique_slices = np.unique(z_indices)

            for z_slice in unique_slices:
                # 1. 2D 원본 단면 슬라이스 추출
                raw_slice = ct_arr[z_slice, :, :]
                mask_slice = seg_arr[z_slice, :, :] == label_id  # 이진 마스크 분리

                # 2. HU 윈도잉 연산 및 [0, 1] 정규화 처리 (마스크 제외)
                windowed_slice = apply_lung_window(raw_slice)

                # 3. 고정 중심점 기준 Crop 단면 추출
                # CT 영상은 0.0(MIN_HU 공기값), 마스크는 배경값 0으로 여백 처리
                patch_ct = safe_crop_2d(
                    windowed_slice, crop_center_y, crop_center_x, crop_size=CROP_SIZE, pad_value=0.0
                )
                patch_mask = safe_crop_2d(
                    mask_slice.astype(np.uint8), crop_center_y, crop_center_x, crop_size=CROP_SIZE, pad_value=0
                )

                ct_patch_filename = f"nodule_{nodule_idx}_slice_{z_slice}_ct.npy"
                mask_patch_filename = f"nodule_{nodule_idx}_slice_{z_slice}_mask.npy"

                ct_patch_path = subj_patch_dir / ct_patch_filename
                mask_patch_path = subj_patch_dir / mask_patch_filename

                np.save(ct_patch_path, patch_ct)
                np.save(mask_patch_path, patch_mask)

                manifest_records.append({
                    "subject_id": subject_id,
                    "nodule_idx": nodule_idx,
                    "slice_z": z_slice,
                    "center_y": crop_center_y,
                    "center_x": crop_center_x,
                    "avg_malignancy": avg_malignancy,
                    "raw_malignancy": str(flat_malignancy),
                    "ct_patch_path": str(ct_patch_path),
                    "mask_patch_path": str(mask_patch_path),
                })

    # 데이터프레임 구조화 후 CSV 백업 저장
    if manifest_records:
        df_manifest = pd.DataFrame(manifest_records)
        manifest_csv_path = PATCHES_ROOT / "patch_manifest.csv"
        df_manifest.to_csv(manifest_csv_path, index=False, encoding="utf-8")
        logger.info(f"마스터 매니페스트 파일 빌드 완료 ➔ 총 {len(df_manifest)}개 슬라이스 패치 등록")
        logger.info(f"저장소 위치: {manifest_csv_path}")
    else:
        logger.warning("생성된 유효 패치 데이터 엔트리가 없습니다.")


if __name__ == "__main__":
    build_patch_dataset()
