# src/preprocessing/match_raw.py
#
# ─── 역할 ─────────────────────────────────────────────────────────────
# - parse_lidc_annotations + match_dicom 통합
# - Extract UIDs of each nodule slices
#      → Save to .json file
#
# ─── 파이프라인 상 위치 ─────────────────────────────────────────────────
# [1] [현재] match_raw.py
# [2] nodule_segment.py
# [3] npy_crop.py
# [4]
#
# ─── 출력 ──────────────────────────────────────────────────────────────
# data/processed/nodule_info.json
#
# ─── 실행 ──────────────────────────────────────────────────────────────
# python -m src.preprocessing.match_raw

import os
import glob
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import logging
import pydicom
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from src.configs.config import (
    SERVER_DICOM_ROOT,
    XML_DIR,
    METADATA_CSV,
    JSON_PATH,
    NODULE_XY_THR,
    NODULE_Z_THR,
    MIN_POLY_PTS,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
NS = {"ns": "http://www.nih.gov"}


# ──────────────────────────────────────────────────────────────────────
# 1. XML 데이터 추출
# ──────────────────────────────────────────────────────────────────────
def build_uid_map(meta_csv: Path) -> dict:
    """
    metadata.csv 로드 -> {series_uid: {subject_id, file_location}} dict 반환.
    """
    df = pd.read_csv(meta_csv)
    return {
        str(row["Series UID"]).strip(): {
            "subject_id": str(row["Subject ID"]).strip(),
            "file_location": str(row["File Location"]).strip(),
        }
        for _, row in df.iterrows()
    }


def collect_raw_nodules(root: ET.Element) -> list:
    """
    xml root에서 판독자 별 결절 후보를 flat list로 수집
    [변경] imageZposition에 기술된 값 대신 고유 식별자인 imageSOP_UID 추출

    제외 조건:
        - characteristics or malignancy tag 없음
        - malignancy 값이 1~5 범위 밖
        - polygon edge 수 < MIN_POLY_PTS
        - ROI 정보가 없음

    Returns (for each nodules):
        {
            "reader_id": int,  # 판독자 idx
            "malignancy: int,  # 악성도 점수
            "rois" : [
                {
                    "z_uid": str,  # imageSOP_UID (슬라이스 고유 식별자)
                    "polygon": [{"x": float, "y": float}, ...],
                }, ...
            ]
        }
    """
    raw_nodules = []
    for reader_id, session in enumerate(root.findall(".//ns:readingSession", NS)):
        for nodule in session.findall("ns:unblindedReadNodule", NS):
            # collect Malignancy
            char = nodule.find("ns:characteristics", NS)
            if char is None:
                continue
            mal_elem = char.find("ns:malignancy", NS)
            if mal_elem is None or not mal_elem.text:
                continue
            malignancy = int(mal_elem.text)
            if malignancy not in range(1, 6):
                continue

            # collect ROIs
            rois = []
            for roi in nodule.findall("ns:roi", NS):
                z_elem = roi.find("ns:imageSOP_UID", NS)
                if z_elem is None:
                    continue
                if z_elem.text is None:
                    continue

                polygon = []
                for edge in roi.findall("ns:edgeMap", NS):
                    x_elem = edge.find("ns:xCoord", NS)
                    y_elem = edge.find("ns:yCoord", NS)
                    if x_elem is None or y_elem is None or x_elem.text is None or y_elem.text is None:
                        continue
                    x, y = float(x_elem.text), float(y_elem.text)
                    polygon.append({"x": x, "y": y})

                if len(polygon) < MIN_POLY_PTS:
                    continue

                rois.append({"z_uid": z_elem.text.strip(), "polygon": polygon})
            if not rois:
                continue

            raw_nodules.append(
                {
                    "reader_id": reader_id,
                    "malignancy": malignancy,
                    "rois": rois,
                }
            )
    return raw_nodules


# ──────────────────────────────────────────────────────────────────────
# 2. DICOM 데이터 추출
# ──────────────────────────────────────────────────────────────────────
def resolve_dicom_dir(server_root: Path, dicom_path: str) -> Optional[str]:
    """
    metadata.csv > File Location(=dicom_path)

    Return: 실제(DICOM) 폴더 절대경로
    """
    relative = dicom_path.lstrip("./").lstrip(".\\").replace("\\", "/")
    dicom_dir = os.path.join(server_root, relative)
    if os.path.isdir(dicom_dir):
        return dicom_dir
    return None


def build_dicom_spatial_map(dicom_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    DICOM 폴더의 첫 번째 .dcm 헤더에서 공간 정보 추출
    [변경]

    Returns:
        dicom_uid_map:
        sorted_slices:
    """
    dicom_uid_map = {}
    slice_list = []

    for root, _, files in os.walk(dicom_dir):
        for file in files:
            fpath = Path(root) / file
            try:
                dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
                z_uid = str(dcm.SOPInstanceUID).strip()
                if "ImagePositionPatient" in dcm:
                    z_position = float(dcm.ImagePositionPatient[2])
                elif "SliceLocation" in dcm:
                    z_position = float(dcm.SliceLocation)
                else:
                    z_position = float(dcm.InstanceNumber) if "InstanceNumber" in dcm else 0.0
                instance_num = int(dcm.InstanceNumber) if "InstanceNumber" in dcm else 0
                slice_info = {
                    "z_uid": z_uid,
                    "z_position": z_position,
                    "file_path": str(fpath),
                    "instance_number": instance_num,
                }
                dicom_uid_map[z_uid] = slice_info
                slice_list.append(slice_info)
            except (Exception, AttributeError):
                continue
    sorted_slices = sorted(slice_list, key=lambda x: x["z_position"])
    return dicom_uid_map, sorted_slices


def match_dicom(raw_nodules: List[Dict[str, Any]], dicom_uid_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    (설명 추가 예정)
    """
    matched_nodules = []
    for nodule in raw_nodules:
        matched_rois = []
        for roi in nodule["rois"]:
            z_uid = roi["z_uid"]
            if z_uid in dicom_uid_map:
                dicom_info = dicom_uid_map[z_uid]

                updated_roi = {
                    "z_uid": z_uid,
                    "z_position": dicom_info["z_position"],
                    "dicom_path": dicom_info["file_path"],
                    "polygon": roi["polygon"],
                }
                matched_rois.append(updated_roi)
            else:
                logger.warning(f"XML에 기재된 UID가 DICOM 세트에 없음 (skip): {z_uid}")
                continue
        if matched_rois:
            z_positions = [r["z_position"] for r in matched_rois]
            center_z = sum(z_positions) / len(z_positions)  # 각 판독자별 center_z라 평균 처리함
            matched_nodule = {
                "reader_id": nodule["reader_id"],
                "malignancy": nodule["malignancy"],
                "center_z": center_z,
                "rois": matched_rois,
            }
            matched_nodules.append(matched_nodule)
    return matched_nodules


# ──────────────────────────────────────────────────────────────────────
# 3. centroid 거리 기반 결절 매칭
# ──────────────────────────────────────────────────────────────────────
def match_nodules(
    matched_raw_nodules: List[Dict[str, Any]], NODULE_XY_THR: float = NODULE_XY_THR, NODULE_Z_THR: float = NODULE_Z_THR
) -> List[Dict[str, Any]]:
    """
    1. 판독자 간 centroid 거리가 임계값 이내이면 같은 결절로 묶음.
       O(n²) greedy 매칭 — 결절 수가 수십 개 수준이므로 충분히 빠름.
    2. 도출된 rois에서 파생값 계산

    매칭 조건:
      xy 유클리드 거리 ≤ NODULE_XY_THR (픽셀)
      z 거리          ≤ NODULE_Z_THR  (mm)
      같은 판독자끼리는 병합 금지 (판독자 5명 버그 방지)

    Returns (for each nodules):
      {
          'nodule_idx': int,
          'malignancy': [int, ...],   # 판독자별 점수 리스트
          'rois'      : {
              'reader_0': [{'z_position': float, 'polygon': [...]}, ...],
              'reader_1': [...],
              ...
          },
      }
    """
    # 판독자별 결절 후보의 2D Centroid 계산
    prepared_nodules = []
    for raw in matched_raw_nodules:
        all_x = [pt["x"] for roi in raw["rois"] for pt in roi["polygon"]]
        all_y = [pt["y"] for roi in raw["rois"] for pt in roi["polygon"]]
        if not all_x or not all_y:
            continue

        # 2D centroid
        cx = float(np.mean(all_x))
        cy = float(np.mean(all_y))
        cz = raw["center_z"]
        prepared_nodules.append({**raw, "centroid_x": cx, "centroid_y": cy, "centroid_z": cz})

    n = len(prepared_nodules)
    used = [False] * n
    groups = []

    for i in range(n):
        if used[i]:
            continue

        group = [prepared_nodules[i]]
        used[i] = True

        for j in range(i + 1, n):
            if used[j]:
                continue

            # 같은 판독자가 이미 그룹에 있으면 병합 금지
            existing_readers = {m["reader_id"] for m in group}
            if prepared_nodules[j]["reader_id"] in existing_readers:
                continue

            dx = prepared_nodules[i]["centroid_x"] - prepared_nodules[j]["centroid_x"]
            dy = prepared_nodules[i]["centroid_y"] - prepared_nodules[j]["centroid_y"]
            xy_dist = np.sqrt(dx**2 + dy**2)
            dz = abs(prepared_nodules[i]["centroid_z"] - prepared_nodules[j]["centroid_z"])

            if xy_dist <= NODULE_XY_THR and dz <= NODULE_Z_THR:
                group.append(prepared_nodules[j])
                used[j] = True

        groups.append(group)

    nodule_list = []
    for idx, group in enumerate(groups):
        malignancy_list = [m["malignancy"] for m in group]
        rois_by_reader = {}
        for m in group:
            # XML에서 파싱했던 고유 식별자(m["reader_id"])를 그대로 Key로 채택
            reader_key = f"reader_{m['reader_id']}"
            rois_by_reader[reader_key] = [
                {
                    "z_uid": roi["z_uid"],
                    "z_position": roi["z_position"],
                    "dicom_path": roi["dicom_path"],
                    "polygon": roi["polygon"],
                }
                for roi in m["rois"]
            ]

        # compute_derived() 기능 통합
        # 가장 많이 등장한 z값 선택 (동률 시 중앙값)
        z_set = set()
        count_by_z = {}
        for m in group:
            for roi in m["rois"]:
                z = roi["z_position"]
                z_set.add(z)
                count_by_z[z] = count_by_z.get(z, 0) + 1
        all_z = sorted(list(z_set))
        if not all_z:
            continue

        max_count = max(count_by_z.values())
        candidates = [z for z, c in count_by_z.items() if c == max_count]
        raw_center_z = float(np.median(candidates))

        # all_z에서 center_z와 가장 가까운 실제 z 찾기
        closest_z = min(all_z, key=lambda z: abs(z - raw_center_z))
        center_idx = all_z.index(closest_z)

        # center_z 슬라이스에서 꼭짓점 수 최다 polygon 선정
        best_polygon = []
        best_count = -1
        for m in group:
            for roi in m["rois"]:
                if roi["z_position"] != closest_z:
                    continue
                if len(roi["polygon"]) > best_count:
                    best_count = len(roi["polygon"])
                    best_polygon = roi["polygon"]

        if best_polygon:
            center_x = float(np.mean([pt["x"] for pt in best_polygon]))
            center_y = float(np.mean([pt["y"] for pt in best_polygon]))
        else:
            center_x, center_y = 0.0, 0.0

        nodule_list.append(
            {
                "nodule_idx": idx,
                "series_uid": None,  # build_patient_dict에서 채워짐
                "file_location": None,  # build_patient_dict에서 채워짐
                "malignancy": malignancy_list,
                "rois": rois_by_reader,
                "derived": {
                    "all_z_positions": all_z,
                    "num_slices": len(all_z),
                    "center_z": closest_z,
                    "center_slice_idx": center_idx,
                    "center_x": round(center_x, 2),
                    "center_y": round(center_y, 2),
                    "rep_polygon": best_polygon,
                },
            }
        )
    return nodule_list


# ──────────────────────────────────────────────────────────────────────
# 4. 전체 xml 순회 - patient_dict 구성
# ──────────────────────────────────────────────────────────────────────
def build_patient_dict() -> dict:
    """ """
    uid_map = build_uid_map(METADATA_CSV)
    logger.info(f"[1/4] metadata 로드: {len(uid_map)}개 UID")
    xml_paths = sorted(glob.glob(os.path.join(XML_DIR, "**", "*.xml"), recursive=True))
    logger.info(f"[2/4] XML 파일 수집: {len(xml_paths)}개")

    patient_dict = {}
    processed_uids = set()
    skip_no_uid = skip_no_match = skip_no_nodule = skip_duplicate = skip_no_dicom = 0

    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        uid_elem = root.find(".//ns:SeriesInstanceUid", NS)
        if uid_elem is None or uid_elem.text is None:
            skip_no_uid += 1
            continue
        series_uid = uid_elem.text.strip()

        if series_uid in processed_uids:
            skip_duplicate += 1
            continue

        if series_uid not in uid_map:
            skip_no_match += 1
            continue

        processed_uids.add(series_uid)

        subject_id = uid_map[series_uid]["subject_id"]
        file_location = uid_map[series_uid]["file_location"]

        dicom_series_dir = resolve_dicom_dir(SERVER_DICOM_ROOT, file_location)

        if not dicom_series_dir or not os.path.exists(dicom_series_dir):
            skip_no_dicom += 1
            logger.warning(f"DICOM series 원본 폴더 누락 (skip): {dicom_series_dir}")
            continue

        dicom_uid_map, _ = build_dicom_spatial_map(Path(dicom_series_dir))
        if not dicom_uid_map:
            skip_no_dicom += 1
            continue

        raw_nodules = collect_raw_nodules(root)
        if not raw_nodules:
            skip_no_nodule += 1
            continue

        matched_raw_nodules = match_dicom(raw_nodules, dicom_uid_map)
        if not matched_raw_nodules:
            skip_no_nodule += 1
            continue

        processed_uids.add(series_uid)

        nodule_list = match_nodules(matched_raw_nodules)
        for nodule in nodule_list:
            nodule["series_uid"] = series_uid
            nodule["file_location"] = file_location

        if subject_id not in patient_dict:
            patient_dict[subject_id] = {
                "subject_id": subject_id,
                "series_uids": [],
                "nodules": [],
            }

        if series_uid not in patient_dict[subject_id]["series_uids"]:
            patient_dict[subject_id]["series_uids"].append(series_uid)

        patient_dict[subject_id]["nodules"].extend(nodule_list)

    # nodule_idx 재부여 (다중 시리즈 환자의 결절이 합쳐진 후 0부터 순서대로)
    for patient in patient_dict.values():
        for i, nodule in enumerate(patient["nodules"]):
            nodule["nodule_idx"] = i

    logger.info("[3/4] 처리 완료")
    logger.info(f"    - XML 총계         : {len(xml_paths)}개")
    logger.info(f"    - UID 없음         : {skip_no_uid}개")
    logger.info(f"    - 중복 XML 스킵    : {skip_duplicate}개")
    logger.info(f"    - 매핑 실패        : {skip_no_match}개")
    logger.info(f"    - 결절 없는 시리즈 : {skip_no_nodule}개")
    logger.info(f"    - 환자 수          : {len(patient_dict)}명")
    logger.info(f"    - 총 결절 수       : {sum(len(p['nodules']) for p in patient_dict.values())}개")

    return patient_dict


def save_json(patient_dict: dict, json_path: Path) -> None:
    """patient_dict를 subject_id 기준 정렬 후 JSON으로 저장."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    sorted_dict = dict(sorted(patient_dict.items()))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"[4/4] JSON 저장: {json_path}")


if __name__ == "__main__":
    patient_dict = build_patient_dict()
    save_json(patient_dict, JSON_PATH)
