# src/utils/seed.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   실험 재현성(reproducibility)을 위한 random seed 고정
#
# ─── 왜 여러 라이브러리에 모두 seed를 설정하나? ──────────────────────────────
#   Python, NumPy, PyTorch는 각각 독립적인 random 엔진을 가짐.
#   하나만 고정해도 다른 곳의 랜덤성이 남아 실험마다 결과가 달라짐.
#   CUDA의 경우 비결정적 연산(cudnn.benchmark)을 끄면 느려지지만 재현됨.

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    모든 랜덤 엔진에 동일한 seed 설정.

    Args:
        seed: 정수 seed 값 (기본 42 — 관습적 기본값)
    """
    random.seed(seed)              # Python 표준 라이브러리 random
    np.random.seed(seed)           # NumPy random
    torch.manual_seed(seed)        # PyTorch CPU random
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random (다중 GPU 포함)

    # cudnn: GPU 연산의 결정적 실행 강제
    # benchmark=False → 매번 동일한 알고리즘 선택 (재현성 ↑, 속도 약간 ↓)
    # deterministic=True → 비결정적 알고리즘 비활성화
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True

    # 환경 변수: Python 해시 랜덤화 비활성화
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f'[seed] 고정 완료: seed={seed}')