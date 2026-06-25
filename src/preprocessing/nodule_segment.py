# src/preprocessing/nodule_segment.py
#
# ─── 역할 ─────────────────────────────────────────────────────────
# - 결절 영역 segment masking (합의 기반)
#   - 중간에 단절 일어나는 일 방지함
# - 리샘플링 (1mm)
# - nifti 파일로 변환
#
# ─── 파이프라인 상 위치 ─────────────────────────────────────────────────
# [1] match_raw.py
# [2] [현재] nodule_segment.py
# [3] npy_crop.py
#
# ─── 출력 ─────────────────────────────────────────────────────────────
#
#
# ─── 실행 ─────────────────────────────────────────────────────────────
# python -m src.preprocessing.nodule_segment
# nohup python -m src.preprocessing.nodule_segment > output_segment.log 2>&1 &

import os
import sys
import json
import csv
import shutil
import argparse
import logging
from pathlib import Path
from typing import Optional, Any, Mapping, cast
from concurrent.futures import ProcessPoolExecutor, as_completed

import SimpleITK as sitk
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.draw import polygon as sk_polygon
from tqdm import tqdm

from src.configs.config import SERVER_DICOM_ROOT, JSON_PATH, NIFTI_SAVE_DIR, MIN_POLY_PTS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

NUM_WORKERS = os.cpu_count()


# ──────────────────────────────────────────────────────────────────────
# 1. 등방성 리샘플링
# ──────────────────────────────────────────────────────────────────────
def resolve_dicom_dir(server_root: Path, dicom_path: str) -> str:
    """metadata.csv의 File Location 경로를 기반으로 실제 DICOM 폴더 경로 반환"""
    relative = dicom_path.lstrip("./").lstrip(".\\").replace("\\", "/")
    return os.path.join(server_root, relative)


def resample_volume(
    image: sitk.Image, new_spacing: tuple = (1.0, 1.0, 1.0), interpolator=sitk.sitkLinear
) -> sitk.Image:
    """SimpleITK volume -> 1mm3 간격 리샘플링"""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    if orig_spacing == new_spacing:
        return image

    new_size = [int(round(orig_size[i] * orig_spacing[i] / new_spacing[i])) for i in range(3)]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())

    return resample.Execute(image)


# ──────────────────────────────────────────────────────────────────────
# 2. 2D 폴리곤 마스크 드로잉
# ──────────────────────────────────────────────────────────────────────
def draw_reader_mask(resampled_ct: sitk.Image, rater_rois: list, orig_origin: tuple, orig_spacing: tuple):
    """
    원본 픽셀 좌표 → 물리 mm 좌표 변환 후 세그먼트 마스크 생성
    """
    res_size = resampled_ct.GetSize()
    # GetSize 좌표는 (X, Y, Z) 순서 / np.array는 (Z, Y, X) 순서 → 위치 조정
    mask_arr = np.zeros((res_size[2], res_size[1], res_size[0]), dtype=np.uint8)
    has_polygon = False

    for roi in rater_rois:
        polygon = roi.get("polygon", [])
        if len(polygon) < MIN_POLY_PTS:
            continue

        z_position = roi["z_position"]
        res_xs, res_ys, res_zs = [], [], []

        for pt in polygon:
            x_coords_mm = orig_origin[0] + pt["x"] * orig_spacing[0]
            y_coords_mm = orig_origin[1] + pt["y"] * orig_spacing[1]

            try:
                res_idx = resampled_ct.TransformPhysicalPointToIndex((x_coords_mm, y_coords_mm, z_position))
                res_xs.append(res_idx[0])
                res_ys.append(res_idx[1])
                res_zs.append(res_idx[2])
            except Exception:
                continue

        if not res_zs:
            continue

        target_z = int(round(np.mean(res_zs)))
        if not (0 <= target_z < res_size[2]):
            continue

        try:
            rr, cc = sk_polygon(res_ys, res_xs, shape=(res_size[1], res_size[0]))
            mask_arr[target_z, rr, cc] = 1
            has_polygon = True
        except Exception as e:
            logger.error(f"polygon 생성 실패 (skip): {e}")
            continue

    return mask_arr if has_polygon else None


# ──────────────────────────────────────────────────────────────────────
# 3. 메인 파이프라인
# ──────────────────────────────────────────────────────────────────────
def process_single_patient(subject_id: str, patient_info: dict) -> bool:
    """한 명의 환자를 처리하는 독립 함수 (Process-safe)"""
    try:
        series_groups = {}
        for nodule in patient_info.get("nodules", []):
            s_uid = nodule["series_uid"]
            series_groups.setdefault(s_uid, []).append(nodule)

        for s_uid, nodules_in_series in series_groups.items():
            if len(series_groups) > 1:
                series_out_dir = NIFTI_SAVE_DIR / subject_id / s_uid
            else:
                series_out_dir = NIFTI_SAVE_DIR / subject_id

            # 이미 변환이 완료된 환자라면 연산 스킵 (중단 후 재시작 기능 지원)
            if (series_out_dir / "ct.nii.gz").exists() and (series_out_dir / "seg.nii.gz").exists():
                return True

            series_out_dir.mkdir(parents=True, exist_ok=True)
            sample_nodule = nodules_in_series[0]
            dicom_series_dir = resolve_dicom_dir(SERVER_DICOM_ROOT, sample_nodule["file_location"])

            if not os.path.exists(dicom_series_dir):
                return False

            # 1. DICOM 볼륨 로드
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_dir)
            reader.SetFileNames(dicom_names)
            raw_ct = reader.Execute()

            # 2. CT 볼륨 1mm 리샘플링
            resampled_ct = resample_volume(raw_ct, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
            final_seg_arr = np.zeros(resampled_ct.GetSize()[::-1], dtype=np.uint8)
            has_any_seg = False

            orig_origin = raw_ct.GetOrigin()
            orig_spacing = raw_ct.GetSpacing()

            for nodule in nodules_in_series:
                label_id = nodule["nodule_idx"] + 1

                # 의사들의 마스크 투표 점수를 누적할 어레이판 세팅
                nodule_vote_arr = np.zeros(final_seg_arr.shape, dtype=np.int32)
                active_readers = 0

                # 3. 각 의사 마스크를 1mm 공간 위에 드로잉 후 합산 누적
                for _, rater_rois in nodule.get("rois", {}).items():
                    reader_mask_arr = draw_reader_mask(resampled_ct, rater_rois, orig_origin, orig_spacing)
                    if reader_mask_arr is not None:
                        nodule_vote_arr += reader_mask_arr
                        active_readers += 1

                if active_readers == 0:
                    continue

                # 참여 평가자가 2명 이상이면 2인 이상 합의(>=2), 혼자 판독한 결절이면 >=1 적용
                thresh = 2 if active_readers >= 2 else 1
                consensus_arr = (nodule_vote_arr >= thresh).astype(np.uint8)

                if np.any(consensus_arr > 0):
                    # 약식 병합으로 층간이 미세하게 깨지는 현상을 방어하기 위해 3D 후처리만 가볍게 수행
                    consensus_img = sitk.GetImageFromArray(consensus_arr)
                    consensus_img.CopyInformation(resampled_ct)

                    # 3D 구멍 메우기 및 Z축 종방향 닫기 연산 ([1, 1, 2] 커널로 층간 끊김 해결)
                    filled_mask = sitk.BinaryFillhole(consensus_img)
                    closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
                    closing_filter.SetKernelType(sitk.sitkBall)
                    closing_filter.SetKernelRadius([1, 1, 2])
                    closing_filter.SetForegroundValue(1)

                    refined_nodule_mask = closing_filter.Execute(filled_mask)

                    nodule_arr = sitk.GetArrayFromImage(refined_nodule_mask)
                    final_seg_arr[nodule_arr > 0] = label_id
                    has_any_seg = True

            # 파일 저장
            sitk.WriteImage(resampled_ct, str(series_out_dir / "ct.nii.gz"), useCompression=True)
            if has_any_seg:
                final_seg_image = sitk.GetImageFromArray(final_seg_arr)
                final_seg_image.CopyInformation(resampled_ct)
                sitk.WriteImage(final_seg_image, str(series_out_dir / "seg.nii.gz"), useCompression=True)
        return True
    except Exception as e:
        logger.error(f"환자 {subject_id} 변환 실패: {e}")
        return False


def run_nifti_generation():
    if not os.path.exists(JSON_PATH):
        logger.error(f"기반 JSON 파일이 존재하지 않습니다: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        patient_dict = json.load(f)

    logger.info(f"NIfTI 변환 프로세스 기동 (병렬 코어 개수: {NUM_WORKERS})")

    success_count = 0
    # ProcessPoolExecutor를 이용해 환자 리스트를 여러 CPU 코어에 분배하여 동시 처리
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_patient, sid, pinfo): sid for sid, pinfo in patient_dict.items()}

        # tqdm 바를 이용해 병렬 처리 현황 실시간 모니터링
        for future in tqdm(as_completed(futures), total=len(futures), desc="병렬 NIfTI 변환 중"):
            sid = futures[future]
            if future.result():
                success_count += 1

    logger.info(f"전체 작업 완료 -> 성공: {success_count}/{len(patient_dict)}명, 저장소: {NIFTI_SAVE_DIR}")


if __name__ == "__main__":
    run_nifti_generation()
