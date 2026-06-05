# src/utils/device.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   GPU/CPU 디바이스 자동 선택

import torch


def get_device() -> torch.device:
    """
    사용 가능한 디바이스 자동 선택 및 정보 출력.

    우선순위: CUDA(NVIDIA GPU) > MPS(Apple Silicon) > CPU

    Returns:
        torch.device 인스턴스
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)   # 첫 번째 GPU 이름
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'[device] CUDA 사용: {gpu_name} ({gpu_mem:.1f}GB)')

    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2 Mac)용 GPU
        device = torch.device('mps')
        print('[device] MPS 사용 (Apple Silicon)')

    else:
        device = torch.device('cpu')
        print('[device] CPU 사용 (GPU 없음 — 학습 느릴 수 있음)')

    return device