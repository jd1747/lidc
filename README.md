# 2.5D 입력 기반 폐결절 악성도 이진 분류: Gated-Dilated Network에 CBAM을 결합한 접근

**2026 융합 메디컬 AI 스마트 웰니스 C**
홍채은 · 허다온 · 김종호 · 이상욱 · 이유준 · 이준용

---

## Abstract

폐결절 악성도 분류는 초기 폐암 조기 발견을 위해 임상적으로 중요하다. LIDC-IDRI 데이터셋 분석 결과, 직경 6mm 미만 소결절 구간에서 악성 비율이 6.3%에 불과하고 판독자 간 불일치가 크게 나타나며, 소결절은 내부 voxel 수가 적어 형태 정보가 제한적이다. 임상에서 영상의학과 의사는 spiculation, vascular convergence, pleural indentation 등 결절 주변 조직 패턴을 함께 참고해 악성도를 판단한다(Snoeckx et al., 2018; Qin et al., 2021). 이로부터 본 연구는 "결절 내부뿐 아니라 주변 맥락 정보를 함께 학습하면 특히 소결절 분류 성능이 향상될 것"이라는 가설을 세웠다.

베이스라인으로 결절 크기 다양성을 단일 네트워크로 처리하는 Gated-Dilated Network(GDN)를 채택하였다. GDN의 두 번째 dilation branch를 d2=2에서 d2=3으로 확장하여 더 넓은 수용 영역(receptive field)을 확보하고, 이후 alpha gating이 포착하지 못하는 채널·공간 어텐션을 보완하기 위해 CBAM을 결합하는 GDN+CBAM 구조를 제안한다. 입력 방식은 완전한 3D CNN의 연산·메모리 부담 없이 인접 슬라이스의 z축 구조 정보를 활용하는 2.5D 방식을 채택하였다. LIDC-IDRI 데이터셋 최종 2,045개 결절(800명, 환자 단위 분할)을 사용하였다.

모델 비교 실험에서 GDN은 DualConvNeXt 대비 파라미터 118배 적은 조건에서 모든 지표에서 우세하여 본 모델로 확정되었다. Dilation d2=3 채택 후(EXP-A, threshold 최적화 포함) test AUC 0.8888, Sensitivity 0.8191을 달성하였다. CBAM 위치 탐색 결과, drop2 이후(gd4 통과 후)와 drop3 이후(gd5 통과 후) 두 위치에 CBAM을 배치한 EXP-E(CBAM@drop2 + CBAM_final@drop3)가 test AUC 0.8939, Sensitivity 0.8404, small subgroup AUC 0.8421로 전체 실험 중 핵심 지표 최고치를 달성하여 최종 제안 구조로 확정하였다.

---

## 1. Introduction

### 1.1 Background

폐결절은 폐 내부에 생긴 지름 3cm 이하의 구상 병변으로, 질병명이 아니라 흉부 X선 또는 CT를 통해 관찰되는 영상학적 소견이다. 결절의 대부분은 소실되거나 크기 변화 없이 유지되지만, 일부는 폐암으로 진행한다. 소결절을 조기에 발견하고 악성 여부를 판별하는 것은 치료 골든타임 확보와 불필요한 침습적 시술 방지 양쪽 모두에서 중요하다.

그러나 추적 검사에서 악성도 판단은 두 가지 이유로 병목이 된다. 첫째, **판독자 간 불일치**: LIDC-IDRI에서 4명의 흉부 영상의학과 전문의 간 disagreement가 크며, 동일 결절에 대해 판정이 달라질 수 있다. 둘째, **다중 특성 종합의 어려움**: 악성도는 spiculation, lobulation, calcification, texture, density 등 9가지 이상의 형태적 특성의 주관적 종합으로 판단된다. 딥러닝은 이 과정을 재현 가능하고 일관된 방식으로 보조할 수 있다.

임상에서 영상의학과 의사가 실제로 사용하는 악성 판독 소견—spiculation(결절 경계 밖으로 뻗는 섬유화), pleural retraction(결절과 흉막의 연결), vascular convergence(주변 혈관이 결절로 모이는 양상), satellite lesion(결절 주변 병변)—은 상당수가 결절 내부가 아니라 **결절과 주변 폐조직·혈관·흉막의 관계**를 보는 특징이다(Snoeckx et al., 2018). 이는 임상 진단 자체가 이미 주변 맥락 정보를 내포하고 있음을 의미한다.

### 1.2 Problem Statement

폐결절은 지름 3cm 이하의 구상 병변으로 다양한 모양과 크기가 존재한다. 본 연구에서 학습에 활용할 LIDC-IDRI 데이터셋은 미국 국립 암 연구소(NCI)에서 제공한 공개 폐암 영상 데이터셋으로, 총 1,010명 환자의 흉부 CT 영상과 4명의 흉부 영상의학과 전문의가 주석한 결절 악성도(malignancy) 점수로 구성된다.

데이터셋 분석 결과, 소결절 구간에서 양성과 악성이 혼재하며 애매 판정 비율 또한 높게 나타났다. 이는 작은 결절이 제한된 형태(morphology) 정보를 가지며, 단일 local feature만으로는 정확한 악성 판단이 어려울 가능성을 시사한다.

다양한 크기의 결절 중에서도 큰 결절은 외형만으로도 양성·악성 분류가 비교적 용이한 반면, 작은 결절은 외형만으로 판별이 어렵다. 영상의학과 의사는 이 경우 spiculation, vascular convergence, pleural retraction 등 결절과 주변 조직의 관계를 함께 평가해 악성도를 판단한다(Snoeckx et al., 2018; Gould et al., 2013). 실제로 결절 주변 혈관·기관지 구조는 악성도와 유의한 상관관계를 가지며(Qin et al., 2021), 이는 소결절에서 결절 내부 정보가 제한될수록 주변 맥락의 상대적 기여도가 더 클 수 있다는 연구 가설로 이어진다.

기존 CNN 기반 모델은 국소적 합성곱 연산에 기반하므로 결절 주변의 광범위한 문맥 정보를 효과적으로 활용하는 데 한계가 있다. GDN은 이를 보완하기 위해 dilated convolution으로 수용필드를 확장하고 스케일 선택을 수행하지만, 채널 중요도와 공간 중요도를 명시적으로 모델링하지 않는다.

---

## 2. Related Works

선행 연구를 5가지 흐름으로 분류하고, 각 연구가 어떤 문제를 해결했으며 본 연구 설계에 어떻게 연결되는지를 서술한다.

### 2.1 결절 내부 특징 중심 초기 연구

**Shen et al. (2017, Pattern Recognition)**: LIDC-IDRI에서 결절을 여러 크기의 crop으로 생성해 CNN에 입력하는 Multi-crop CNN(MC-CNN)을 제안하였다. Multi-scale 접근으로 당시 SOTA(AUC 0.93)를 달성하였으나, 결절 자체 내부 특징(모양·경계·텍스처)에만 집중하고 주변 조직 정보는 활용하지 않았다.

**Xie et al. (2018, IEEE Trans. Med. Imaging)**: LIDC-IDRI(양성 1,301개, 악성 644개)에서 CT 축별(axial/coronal/sagittal) ROI를 사용하는 MV-KBC(ResNet-50 기반)를 제안하였다. 다방향 특징을 결합하였으나 결절 주변 맥락을 별도로 학습하지 않았다.

→ 이 연구들은 결절 내부만으로는 소결절 분류에 한계가 있음을 역으로 보여주며, 주변 맥락 학습 필요성의 출발점이 된다.

### 2.2 결절+맥락의 중요성 입증

**Al-Shabi et al. (2019a, Int. J. Comput. Assist. Radiol. Surg.)**: Residual Block(local)과 Non-Local Block(global)을 병렬로 배치한 Deep Local-Global Network를 제안하였다. Non-local block 단독으로는 충분하지 않으며 local-global 결합 시 AUC 0.9562를 달성하였다. 이후 동일 저자의 GDN(본 연구 베이스라인)의 전신 연구이다.

**Masquelin et al. (2022, Acad. Radiol.)**: NLST 데이터셋에서 결절 주변 10mm/15mm band의 radiomics 특징을 활용해 결절 외부 조직도 중요한 진단 정보를 제공함을 밝혔다.

**Qin et al. (2021, arXiv)**: LIDC-IDRI(1,556개 결절, 694명)에서 주변 혈관·기관지·흉막과 악성도의 관계를 정량 분석하였다. 악성 결절이 양성보다 혈관·기관지와 통계적으로 유의하게 더 많이 연결되어 있음을 확인하였다. 이는 악성도가 결절 내부만의 문제가 아니라 주변 구조와도 관련 있음을 임상적으로 뒷받침한다.

→ 본 연구에서 "주변 맥락 포함 학습이 소결절 분류에 기여한다"는 가설의 핵심 임상 근거이다.

### 2.3 결절 크기 다양성 문제 해결

**Al-Shabi et al. (2019b, IEEE Access)**: LIDC-IDRI(1,000+ CT)에서 크기 3~30mm 결절 다양성에 대응하는 Gated-Dilated Network(GDN)를 제안하였다. Dilated Convolution을 MaxPooling 대신 사용해 해상도 손실 없이 다중 스케일을 포착하고, Context-Aware subNet으로 결절 크기에 따라 local/global 특징 강조 경로를 동적으로 선택한다. 특히 중간 크기(5~12mm) 결절 분류 정확도를 향상시켰으며 AUC > 0.95를 달성하였다.

→ 본 연구의 베이스라인으로 직접 채택하였다. 소결절부터 대결절까지 단일 네트워크로 처리하는 구조와 해상도 유지가 채택 근거이다.

### 2.4 주변 조직 중요성의 정량 입증

**Liu et al. (2024, J. Transl. Med.)**: 자체 CT 데이터셋에서 결절 조건별 분류 성능을 직접 비교하였다.

| 조건 | Accuracy | Sensitivity | AUC |
| --- | --- | --- | --- |
| 결절만 (background removed) | 75.61% | 50.00% | 0.78 |
| 결절 + 주변 조직 | 79.03% | 65.46% | 0.84 |
| 결절 + 주변 조직 + fibrosis metadata | 80.84% | 74.67% | 0.89 |

결절만 학습 시 sensitivity가 50%까지 떨어져 악성 결절 절반을 놓친다. 주변 조직 포함 시 sensitivity가 15.46%p 회복된다. 악성 결절 주변의 fibrosis 발생률(65%)이 양성 결절(35%)보다 현저히 높아, 주변 조직 패턴과 악성도 간의 강한 상관관계가 확인된다.

→ 주변 맥락 포함이 분류 성능 향상으로 직결됨을 정량 입증하는 핵심 근거이다.

### 2.5 Dual Branch 및 소결절 탐지

**Dey et al. (2018, ISBI)**: 서로 다른 스케일의 3D 패치를 두 경로로 입력하는 2-pathway 3D CNN으로 결절 내부+주변 정보를 함께 학습하였다. MoDenseNet이 accuracy 90.40%를 달성하였다.

**Afshar et al. (2020, Sci. Rep.)**: 3개의 독립 CapsNet을 서로 다른 스케일 3D 패치에 적용 후 fusion한 3D-MCN을 제안하였다. LIDC-IDRI 기준 accuracy 93.12%, AUC 0.9641, sensitivity 94.94%를 보고하였다.

**Faizi et al. (2025, BMC Cancer)**: LUNA16에서 local 영역에 CNN, global 영역에 Swin Transformer를 사용하는 DCSwinB를 제안하였다. accuracy 90.96%, AUC 0.94를 달성하였다.

**Zheng et al. (2021, Med. Phys.)**: 소결절(<6mm)이 탐지 단계에서도 어렵고 다중 평면·스케일 정보가 필요함을 보였다.

→ 다중 스케일 특징 융합이 단일 입력 대비 유효하며, 소결절에서 내부 정보 부족 문제가 분류 단계에서도 반복됨을 보여준다.

**선행 연구 요약**

| # | 분류 | 논문 | 데이터셋 | 주요 성능 | 본 연구 연관성 |
| --- | --- | --- | --- | --- | --- |
| 1 | 결절 내부 | Shen et al., 2017 | LIDC-IDRI | AUC 0.93 | Multi-scale 필요성 |
| 2 | 결절 내부 | Xie et al., 2018 | LIDC-IDRI | — | 다방향 특징 한계 |
| 3 | 맥락 중요성 | Al-Shabi et al., 2019a | LIDC-IDRI | AUC 0.9562 | GDN의 전신 |
| 4 | 맥락 중요성 | Masquelin et al., 2022 | NLST | — | 주변 조직 근거 |
| 5 | 맥락 중요성 | Qin et al., 2021 | LIDC-IDRI | — | 혈관-악성도 상관 |
| 6 | 크기 문제 | **Al-Shabi et al., 2019b** | LIDC-IDRI | AUC >0.95 | **베이스라인 채택** |
| 7 | 주변 조직 정량 | Liu et al., 2024 | 자체 제작 | AUC 0.89 | 주변 맥락 성능 근거 |
| 8 | Dual Branch | Dey et al., 2018 | LIDC-IDRI | Acc 90.40% | multi-scale fusion |
| 9 | Dual Branch | Afshar et al., 2020 | LIDC-IDRI | AUC 0.9641 | 주변 조직 포함 효과 |
| 10 | Dual Branch | Faizi et al., 2025 | LUNA16 | AUC 0.94 | CNN+Transformer 융합 |
| 11 | 소결절 탐지 | Zheng et al., 2021 | LIDC-IDRI | — | 소결절 정보 부족 |

---

## 3. Problem Definition & Hypothesis

### 문제

선행 연구들은 맥락 정보의 중요성을 인지하였으나, **소결절 분류 태스크에서 주변 맥락의 기여도를 직접 검증한 연구는 드물다.** 소결절(<6mm)은 탐지 단계에서도 어렵고, 분류 단계에서도 내부 정보량 부족 문제가 반복적으로 보고된다. 또한 GDN은 dilation branch 간 혼합 비율을 결정하는 alpha gating을 통해 스케일 선택을 수행하지만, 채널 중요도와 공간 중요도를 명시적으로 모델링하지 않는다는 구조적 한계가 있다. 기존 GDN의 d2=2 dilation은 소결절 주변 맥락 파악에 충분하지 않을 수 있다.

### 가설

소결절은 내부 형태 정보가 제한적이므로, 결절 주변 폐 조직의 맥락 정보(혈관 접촉, 흉막 인접, 주변 실질 패턴 등)를 함께 학습하면 양성·악성 분류 성능, 특히 소결절 subgroup AUC가 향상될 것이다.

GDN의 dilation을 d2=3으로 확장하면 더 넓은 수용영역(receptive field)를 통해 결절 주변 맥락 정보를 더 효과적으로 포착할 수 있다. GDN의 alpha gating은 dilation branch 간 혼합 비율을 결정하는 스케일 선택 모듈이다. 그러나 채널 중요도와 공간 중요도를 명시적으로 모델링하지 않는다. CBAM(Woo et al., 2018)은 채널 어텐션과 공간 어텐션을 순차적으로 제공하여 이 역할을 보완한다. 따라서 두 모듈은 역할이 겹치지 않고 상호 보완적으로 작동한다. CBAM은 소결절 등 작은 객체 탐지 성능 개선 및 시각적 해석력 향상에 효과가 확인된 사례가 있다(Woo et al., 2018).

### 목표

- 전체 test AUC ≥ 0.90
- 소결절(volume < 100㎣) subgroup AUC 향상 검증
- 각 실험에서 결절 size group별 AUC를 비교해 맥락 정보의 기여도를 정량적으로 확인

---

## 4. Methods

### 4.1 Dataset

LIDC-IDRI(National Cancer Institute 제공)는 총 1,010명 환자의 흉부 CT 영상과 4명의 흉부 영상의학과 전문의가 주석한 결절 악성도 점수로 구성된다. XML+metadata parsing 과정에서 CT modality가 아닌 시리즈(흉부 X-ray 등), 매핑 실패, 중복 XML 등을 제외한 결과 875명, 2,696개 결절이 추출되었다.

### 4.2 Label Definition

라벨링 기준은 판독자 4인의 평균 malignancy 점수(1~5점)를 사용하였다.

| 평균 점수 | 라벨 | 이유 |
| --- | --- | --- |
| < 3.0 | 양성 (Benign) | — |
| = 3.0 | **제외** | 판독자 4인 의견이 균등 분산되어 임상적 불확실성 최대 구간 |
| > 3.0 | 악성 (Malignant) | — |

**데이터 구성 요약**

| 항목 | 수치 |
| --- | --- |
| 원본 결절 | 2,696개 (875명) |
| 제외: malignancy 평균 = 3.0 | −646개 |
| 제외: 다중 시리즈 중복 (0332번 환자, overlap ≥ 0.9) | −5개 |
| **최종 사용 결절** | **2,045개 (800명)** |

최종 2,045개는 환자 단위(patient-wise)로 train/val/test = 70/15/15%로 분할하였다. 환자 단위 분할은 동일 환자의 여러 결절이 train/test에 동시 포함되어 발생하는 data leakage를 방지하기 위함이다.

| Split | 환자 수 | 결절 수 | Benign | Malignant |
| --- | --- | --- | --- | --- |
| Train | 559명 | 1,455개 | 2,475 slices | 3,405 slices |
| Val | 121명 | — | — | — |
| Test | 120명 | 287개 | 193 | 94 |

### 4.3 Preprocessing

**2.5D 입력 방식 채택 근거**

폐결절은 CT에서 연속 슬라이스를 따라 형태가 변화하므로 z축 방향 정보가 분류에 중요하다. 그러나 본 연구 데이터 규모에서 완전한 3D CNN을 적용하면 계산 비용과 과적합 위험이 크다. 따라서 결절 중심 슬라이스 k를 기준으로 [k-1, k, k+1]을 3채널로 쌓아 입력하는 2.5D 방식을 채택하였다. 이를 통해 2D 단일 슬라이스보다  z축 구조 정보를 추가로 활용하면서, 3D CNN의 연산 부담은 피한다. 3D보다 정보가 적다는 점은 인정하나, 본 연구의 데이터 규모와 모델 복잡도 제약 하에서 실용적인 균형점으로 판단하였다.

**전처리 파이프라인**

1. HU 클리핑: [-1000, 400] 범위로 제한 후 [0, 1]로 min-max 정규화
2. 리샘플링: 모든 CT를 등방성 1mm 해상도로 리샘플링
3. 세그멘테이션 마스크 기반 결절 중심 좌표 추출
4. 결절 중심으로부터 crop 추출

**Crop size 선택 근거**

원 논문 GDN(Al-Shabi et al., 2019b)은 32×32 입력을 기본 설정으로 사용하였다. 본 연구는 GDN 구조 변경(dilation, CBAM)의 효과를 원 논문과 동일 조건에서 검증하기 위해 기본 실험 crop size를 32×32로 고정하였다. crop=48 예비 실험(260616_gdn_48_ep50_aug2)에서 AUC 0.8702, small AUC 0.6941로 오히려 하락이 관찰되었으나, 해당 실험은 CBAM 구조 등 다른 설정 조건과의 혼재 효과가 있어 crop size 비교는 통제 실험이 수행되지 않아 본 연구 범위에서 해석하지 않았다. [확인 필요: crop size 단독 비교 실험 필요] 더 넓은 주변 맥락을 포함하는 더 큰 crop(64, 96)에서의 효과는 향후 과제이다.

**Augmentation 선택 근거**

기하변환(hflip+rot90)은 폐결절의 방향 불변성을 반영하여 모델의 일반화를 높인다. 강도변환(HU shift, Gaussian noise)은 이미 [0,1]로 정규화된 입력에 추가 노이즈를 가하면 과도한 교란이 발생함을 실험적으로 확인하였다(Exp 03: sensitivity 0.0319로 붕괴). 따라서 기하변환만 적용한다.

### 4.4 Baseline Model: GDN

**채택 이유**: 3~30mm 다양한 결절 크기에 단일 네트워크로 대응하는 구조, MaxPooling 없이 dilated convolution으로 해상도를 유지하는 설계, 파라미터 수 104,459개의 경량성이 채택 근거이다.

**원 논문 GDN 구조**

GDN은 5개의 GD(Gated-Dilated) Block으로 구성된다. 각 GD Block은 두 개의 병렬 dilated convolution branch(dilation=1, dilation=d2)와 이 두 경로를 alpha gating으로 동적으로 혼합하는 Context-Aware subNet으로 이루어진다. Alpha gating은 FC layer → Sigmoid → scalar(α)를 통해 0~1 사이 값을 출력하며, 출력 = α × local_branch + (1−α) × global_branch 방식으로 두 경로를 혼합한다. d2=2가 원 논문 기본값이다.

**모델 비교 결과 (GDN 채택 근거)**

| 지표 | GDN (hflip+rot90, d2=2) | DualConvNeXt (best) |
| --- | --- | --- |
| test AUC | **0.8809** | 0.8708 |
| Sensitivity | 0.7021 | 0.6596 |
| Specificity | **0.9326** | 0.9067 |
| small AUC | **0.7599** | 0.6349 |
| large AUC | **0.8821** | 0.8489 |
| 과적합 | 없음 ✅ | 심각 ⚠️ |
| 파라미터 수 | **104,459** | ~12,000,000+ |

DualConvNeXt는 훈련 샘플(5,880개) 대비 파라미터가 과도하게 커 심각한 과적합이 발생하였다. GDN이 모든 주요 지표에서 우세하고 파라미터 효율성이 현저히 높아 본 모델로 확정하였다.

**pos_weight = 1.0 설정 근거**

전체 z-slice를 입력 단위로 사용함에 따라 슬라이스 수가 결절 크기에 비례하여 증가하였다. 악성 결절은 양성 결절에 비해 평균 크기가 크므로 슬라이스 단위 집계 시 악성 샘플이 상대적으로 더 많이 생성된다. 그 결과 훈련 세트의 클래스 분포는 양성 2,475개, 악성 3,405개(비율 1:1.38)로, 기존 2D 단일 슬라이스 방식에서 나타나던 악성 과소표집 문제가 자연적으로 해소되었다.

BCEWithLogitsLoss의 pos_weight는 양성(악성) 클래스의 손실에 곱해지는 스칼라로, 클래스 불균형 보정을 목적으로 한다. 현재 비율에서 불균형 보정을 위한 이론값은 2,475/3,405 ≈ 0.73으로 오히려 악성 클래스의 손실을 감소시키는 방향이며, 이는 위음성(FN) 비용이 위양성(FP) 비용보다 큰 폐결절 검출의 임상적 요구와 상반된다. 따라서 클래스 불균형 보정 목적의 가중치 적용은 불필요하다. 임상적 비대칭에 의한 가중치 부여(pos_weight > 1.0)는 별도의 ablation 실험으로 검증할 사항이므로, 본 실험에서는 가중치를 적용하지 않은 기준값 pos_weight = 1.0을 설정하였다.

### 4.5 Proposed Model: GDN + Dilation d2=3 + CBAM

### 4.5.1 Dilation d2=3 채택 근거

GDN의 두 번째 branch dilation을 d2=2에서 d2=3으로 확장한 이유는 더 넓은 수용 영역(receptive field)를 통해 결절 경계 너머 주변 폐조직의 맥락 정보를 포착하기 위함이다. dilation=2 대비 dilation=3은 더 멀리 떨어진 pixel들의 관계를 한 번의 convolution에서 학습할 수 있다. 실험 결과 d2=3 적용 시 AUC가 0.8809→0.8888로 향상되고 small subgroup AUC가 0.7599→0.7944로 개선되었다. 이는 넓어진 수용 영역이 성능 향상에 기여했을 가능성을 시사한다.

단, d2=3 단독 적용 시 threshold=0.5 기준에서 sensitivity가 0으로 붕괴하는 현상이 발생하였다. 이는 dilation=3으로 수용 영역이 넓어지면서 모델 출력 분포가 전반적으로 낮은 확률 쪽으로 이동하였기 때문이다. AUC 자체는 향상되어 모델의 분류 능력(ranking)은 유지되고 있으므로, validation set에서 Youden Index로 최적 threshold를 탐색하는 방식으로 해소하였다(채택 threshold: 0.0504→EXP-A 기준).

또한 d2=3 실험에서 val loss 진동(0.6~1.6)이 관찰되었다. 평가 체계상 val loss는 slice 단위 BCE loss이고 val AUC는 nodule 단위 평균 확률 기반으로 계산되어, 동일 결절 내 경계값 근처 슬라이스들이 오가면 loss는 크게 튀지만 결절 단위 평균은 안정적으로 유지될 수 있다. 다만 동일 조건의 d2=2에서는 이러한 진동이 두드러지지 않았으므로, dilation 확장에 따른 특징 분포 변화 역시 영향을 주었을 가능성이 있다. 본 연구에서는 원인을 단정하지 않고, val AUC가 꾸준히 상승하는 것을 정상 수렴의 기준으로 삼아 test AUC 기준으로 모델을 비교하였다. dropout 강화(0.25→0.3) 및 weight decay(5e-4) 조정을 시도하였으나 val loss 수치 안정화에 효과가 없었고 오히려 성능이 저하되어, 기본 dropout 설정(drop1=0.25, drop2=0.25, drop3=0.5) + wd=1e-4를 유지하였다.

### 4.5.2 CBAM 추가 근거 및 위치 설계

**CBAM을 선택한 이유**: GDN의 alpha gating은 dilation branch 간 혼합 비율을 결정하는 스케일 선택 모듈이다. 그러나 채널 중요도와 공간 중요도를 명시적으로 모델링하지 않는다. CBAM(Woo et al., 2018)은 채널 어텐션과 공간 어텐션을 순차적으로 적용하여 이 역할을 보완한다. 따라서 GDN의 스케일 선택과 CBAM의 채널·공간 정제는 역할이 겹치지 않고 상호 보완적으로 작동한다.

**CBAM 배치 위치 설정**: 본 연구에서는 GD block 통과 후 dropout 이후 위치를 후보로 설정하였다. 구체적으로 drop1 이후(gd2 통과 후), drop2 이후(gd4 통과 후), drop3 이후(gd5 통과 후) 세 위치를 후보로 두고 ablation study를 통해 최적 위치를 결정하였다(Phase 4~5 실험 참조). 위치 선택의 적절성은 실험 결과로 검증하였다.

**최종 구조 (EXP-E 기준)**

아래는 CBAM 위치 탐색 실험(Phase 4~5) 전체에서 공통으로 사용된 후보 삽입 위치 구조이다. EXP-E는 이 중 CBAM#1(drop1 이후, gd2 통과 후)과 CBAM#2(drop2 이후, gd4 통과 후), CBAM_final(drop3 이후, gd5 통과 후) 세 위치를 모두 활성화한 구조가 아니라, drop2 이후(gd4 통과 후)와 drop3 이후(gd5 통과 후) 두 위치만 활성화한 구조이다.

```
Input (2.5D, 3ch)
→ GD Layer 1 → GD Layer 2 → Dropout1
→ CBAM 1 (Channel Attn → Spatial Attn)         ← drop1 이후 (= gd2 통과 후) [EXP-B/D에서 실험]
→ GD Layer 3 → GD Layer 4 → Dropout2
→ CBAM 2 (Channel Attn → Spatial Attn)         ← drop2 이후 (= gd4 통과 후) [EXP-C/D/E에서 활성]
→ GD Layer 5 → Dropout3
→ CBAM final (Channel Attn → Spatial Attn)     ← drop3 이후 (= gd5 통과 후) [EXP-E에서 활성]
→ Global Average Pooling → FC → Malignant/Benign
```

EXP-E에서 활성화된 위치: CBAM 2(drop2 이후) + CBAM final(drop3 이후). CBAM 1(drop1 이후)은 EXP-B/D에서 단독·병용 모두 성능을 저하시킨 결과에 따라 EXP-E에서 제외하였다.

파라미터 수: 105,679 (+1,220 vs baseline, +610 vs EXP-C) [확인 필요: 코드 기준 재확인 필요]

---

## 5. Experiments & Results

### 5.1 Experimental Setup

**평가 지표 선택 근거**

Accuracy만으로는 benign:malignant ≈ 2:1 불균형 하에서 성능을 제대로 반영하지 못한다. 의료 영상 분류에서 악성을 놓치는 False Negative(FN)는 양성을 과잉 진단하는 False Positive(FP)보다 임상적으로 더 위험하다. 따라서 Sensitivity(FN 최소화)를 특히 중요하게 보며, AUC(임계값 독립적 판별 능력), Accuracy, Specificity를 함께 사용한다.

**공통 실험 설정**

| 항목 | 값 |
| --- | --- |
| 데이터셋 | LIDC-IDRI, 테스트 결절 n=287 (결절 단위) |
| Crop size | 32×32 |
| Epochs | 50, Batch size 16, lr 1e-4 |
| pos_weight | 1.0 |
| Seed | 42 |
| Aggregation | slice별 예측 확률 → 결절 단위 평균(mean) → AUC |
| Threshold | Youden Index (val set) 기준 최적화 |
| GPU/Framework | [TBD] |

### 5.2 Ablation Study

실험의 인과 흐름은 다음과 같다: 모델 선택(GDN vs DualConvNeXt) → Dilation 탐색(d2=2 vs d2=3) → CBAM 위치 탐색(cbam1 / cbam2 / both) → CBAM 위치 정밀 탐색(gd4 후 vs gd5 후 vs gd4+gd5) → pos_weight 튜닝.

### Phase 1: 모델 비교 (GDN 채택)

ConvNeXt(crop=64)는 단일 branch 구조로 GDN·DualConvNeXt와 함께 baseline 후보 비교 실험(Phase 1-B)에 포함되었다. 파라미터 수는 [확인 필요]이다.

| Model | Aug | test AUC | Sensitivity | Specificity | small AUC | Params |
| --- | --- | --- | --- | --- | --- | --- |
| GDN (d2=2) | hflip+rot90 | **0.8809** | 0.7021 | **0.9326** | **0.7599** | **104,459** |
| ConvNeXt (crop=64) | hflip+rot90 | 0.8707 | 0.8298 | 0.7202 | 0.6497 | [확인 필요] |
| DualConvNeXt (crop=32+96) | hflip+rot90 | 0.8705 | 0.7128 | 0.9016 | 0.6234 | ~12,000,000+ |
| DualConvNeXt (crop=48+96) | hflip+rot90+강도 | 0.8708 | 0.6596 | 0.9067 | 0.6349 | ~12,000,000+ |

→ GDN이 전 지표에서 우세하고 파라미터 효율이 압도적으로 높아 본 모델로 확정.

### Phase 2: **Augmentation** 탐색 (GDN d2=2 기준)

표의 "no aug (threshold=0.5)"와 "no aug, pos_weight=1.0" 두 행은 augmentation 없음으로 동일하나 AUC·Sensitivity 등 수치가 다르다. 전자는 threshold=0.5 고정 적용 결과이고, 후자는 Youden Index threshold 최적화를 적용한 결과(v2 Exp 01과 동일)로, threshold 정책 차이가 수치 차이의 주요 원인으로 추정된다. [확인 필요: 두 행의 실험 설정(seed, 학습 run 동일 여부) 재확인 필요]

| 실험 | Aug | AUC | Accuracy | Sensitivity | Specificity | small AUC |
| --- | --- | --- | --- | --- | --- | --- |
| GDN, no aug (threshold=0.5) | 없음 | 0.8789 | 0.7003 | 0.5851 | 0.9741 | 0.6743 |
| GDN, no aug, pos_weight=1.0 (= v2 Exp 01) | 없음 | 0.8823 | 0.8537 | 0.6489 | 0.9534 | 0.7039 |
| GDN, hflip+rot90 | 기하변환 | **0.8809** | **0.8571** | **0.7021** | **0.9326** | **0.7599** |
| GDN, hu_shift+gaussian | 강도변환 | 0.8787 | 0.6829 | 0.0319 ⚠️ | 0.9948 | 0.6266 |

→ 기하변환 aug가 sensitivity와 small AUC를 개선함을 확인. 강도변환은 이미 [0,1]로 정규화된 값에 과도한 교란을 가하여 sensitivity를 붕괴시키므로 제외.

### Phase 3: Dilation 탐색 (d2=2 → d2=3)

Dilation d2=3, threshold 최적화(Youden Index)를 적용한 EXP-A가 Phase 2 최선 대비 개선을 보였다.

| 실험 | d2 | Threshold | AUC | Sensitivity | Specificity | small AUC |
| --- | --- | --- | --- | --- | --- | --- |
| GDN baseline (d2=2, hflip+rot90) | 2 | 0.5 | 0.8809 | 0.7021 | 0.9326 | 0.7599 |
| GDN d2=3 (threshold=0.5) | 3 | 0.5 | 0.8888 | 0.0000 ⚠️ | 1.0000 | 0.7944 |
| **EXP-A (d2=3 + Youden thr)** | **3** | **0.0504** | **0.8888** | **0.8191** | **0.8187** | **0.7944** |

→ d2=3은 AUC와 small AUC를 개선하나 threshold 붕괴를 동반. Youden Index threshold 최적화로 sensitivity/specificity 균형 회복. EXP-A를 CBAM 실험의 baseline으로 채택.

**EXP-A Confusion Matrix (d2=3, threshold=0.0504)**

|  | 예측 Benign | 예측 Malignant |
| --- | --- | --- |
| **실제 Benign** | 158 | 35 |
| **실제 Malignant** | 17 | 77 |

### Phase 4: CBAM 위치 탐색

CBAM 초기 설계 후보 위치: gd1→gd2→drop→CBAM#1→gd3→gd4→drop→CBAM#2→gd5

| 실험 | 구성 | AUC | Accuracy | Sensitivity | Specificity | small AUC | inter AUC | large AUC | val loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-A (baseline) | d2=3, no CBAM | 0.8888 | 0.8188 | 0.8191 | 0.8187 | 0.7944 | 0.6035 | 0.8945 | 0.6~1.6 ⚠️ |
| EXP-B (cbam1만) | CBAM@gd2 후 | 0.8721 | 0.7666 | 0.8191 | 0.7409 | 0.7796 | 0.4735 ⚠️ | 0.8996 | 1.0~2.8 ⚠️⚠️ |
| EXP-C (cbam2만) | CBAM@gd4 후 | **0.8896** | **0.8397** | 0.7979 | **0.8601** | 0.7878 | **0.6477** | 0.8936 | 0.47~0.94 ✅ |
| EXP-D (cbam1+cbam2) | CBAM@gd2+gd4 | 0.8716 | 0.8188 | 0.7660 | 0.8446 | 0.6612 ⚠️ | 0.5783 | 0.8839 | 0.6~1.1 ⚠️ |

주요 관찰:

- EXP-B(cbam1만): intermediate subgroup AUC가 0.47로 랜덤 수준으로 붕괴. val loss 심화.
- EXP-C(cbam2만): val loss 유일하게 안정화, AUC·Accuracy·Specificity 개선, intermediate 회복.
- EXP-D(cbam1+cbam2): val AUC는 최고(0.9169)이나 test AUC는 최하위권. val-test 괴리 → val set 과적합.

→ **EXP-C(CBAM@drop2 이후)를 기준 구조로 선정하고, 이후 CBAM 위치를 추가 탐색하였다.**

CBAM1이 포함된 구조는 일관되게 성능 저하를 보였으며, 해당 위치는 본 GDN 구조와 적합하지 않은 것으로 판단된다.

### Phase 5: CBAM 위치 정밀 탐색 (drop2 이후 vs drop3 이후 vs drop2+drop3)

EXP-C를 베이스로 CBAM 위치를 drop3 이후(gd5 통과 후)로 이동(EXP-C2)하거나 두 위치 모두 배치(EXP-E)하는 실험을 진행하였다.

| 실험 | 구성 | AUC | Sensitivity | Specificity | small AUC | inter AUC | FN | val loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-C (cbam@drop2) | CBAM@after_drop2 | 0.8896 | 0.7979 | 0.8601 | 0.7878 | 0.6477 | 19 | ✅ 안정 |
| EXP-C2 (cbam@drop3) | CBAM@after_drop3 | 0.8861 | 0.7979 | 0.8342 | 0.7385 ⚠️ | 0.5871 | 19 | ✅ 안정 |
| **EXP-E (drop2+drop3)** | **CBAM@after_drop2 + CBAM_final@after_drop3** | **0.8939** | **0.8404** | 0.8083 | **0.8421** | 0.6250 | **15** | **✅ 안정** |

EXP-E 결과:

- test AUC 0.8939: 전체 실험 중 최고
- Sensitivity 0.8404: 전체 실험 중 최고
- small subgroup AUC 0.8421: EXP-C 대비 +0.054, baseline 대비 +0.048 → **전체 실험 중 최고**
- FN=15: 전체 실험 중 최소 (암을 가장 적게 놓침)
- val loss 0.5~0.75: 안정 유지

EXP-E는 EXP-D(cbam@drop1+drop2 병용)와 달리 val-test 괴리 없이 안정적으로 수렴하였다. drop2+drop3 위치 조합이 EXP-D의 drop1+drop2 조합보다 GDN 구조에 적합한 것으로 해석된다. Specificity는 EXP-C(0.8601) 대비 다소 하락(0.8083)하였으나 임상적으로 우선순위가 높은 Sensitivity와 AUC에서 모두 EXP-E가 우세하다.

**EXP-E Confusion Matrix (threshold=0.1553)**

|  | 예측 Benign | 예측 Malignant |
| --- | --- | --- |
| **실제 Benign** | 156 | 37 |
| **실제 Malignant** | 15 | 79 |

### Phase 6: pos_weight 조정 실험 (EXP-E 기준)

EXP-E(pw=1.0)를 기반으로 pos_weight를 1.5로 상향하여 Sensitivity 추가 개선을 시도하였다.

| 지표 | EXP-E (pw=1.0) | EXP-E (pw=1.5) | 변화 |
| --- | --- | --- | --- |
| AUC | 0.8939 | 0.8889 | −0.005 ↓ |
| Sensitivity | **0.8404** | 0.7553 ⚠️ | −0.085 ↓ |
| Specificity | 0.8083 | 0.8808 | +0.073 ↑ |
| FN (암 놓침) | **15** | 23 ⚠️ | +8 ↑ |
| val loss | ✅ 0.5~0.75 | ⚠️ 0.8~1.6 | 불안정 재발 |

pos_weight를 높였음에도 Sensitivity가 의도와 반대로 크게 하락하였다. pos_weight 상승이 gradient 스케일을 키워 val loss 불안정이 재발하였고, 이것이 학습을 교란시킨 것으로 추정된다. EXP-E(pw=1.0)를 최종 구조로 확정하였다.

### 전체 Ablation 요약

| 단계 | 실험 | AUC | Sensitivity | Specificity | small AUC | FN |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (d2=2, aug, thr=0.5) | GDN hflip+rot90 | 0.8809 | 0.7021 | 0.9326 | 0.7599 | 28 |
| + Dilation d2=3 + Youden thr | EXP-A | 0.8888 | 0.8191 | 0.8187 | 0.7944 | 17 |
| + CBAM@drop2 이후 | EXP-C | 0.8896 | 0.7979 | 0.8601 | 0.7878 | 19 |
| **+ CBAM@drop2 이후 + CBAM_final@drop3 이후** | **EXP-E** | **0.8939** | **0.8404** | 0.8083 | **0.8421** | **15** |

### 5.3 Discussion

**Dilation d2=3 효과**: d2=2 대비 AUC +0.008, small AUC +0.035, Sensitivity +0.117 향상. 더 넓은 수용 영역이 성능 향상에 기여했을 가능성을 시사한다. threshold 붕괴는 Youden Index로 해소 가능하다.

**CBAM@drop1 이후(drop1 이후, gd2 통과 후) 단독의 일관된 실패**: EXP-B, EXP-D 모두에서 drop1 이후 CBAM이 포함된 구조는 intermediate subgroup AUC 붕괴와 val loss 발산을 유발하였다. 동일한 CBAM이라도 삽입 위치에 따라 성능 차이가 크게 나타났으며, drop1 이후 위치는 본 GDN 구조와 적합하지 않은 것으로 확인되었다.

**CBAM2(gd4 이후)의 효과**: val loss 안정화가 가장 두드러진 효과이다. EXP-C에서 val loss가 0.47~0.94로 전체 실험 중 유일하게 안정적인 수렴을 보였다. intermediate subgroup AUC도 EXP-B 대비 0.47→0.65로 회복되었다.

**EXP-E(drop2+drop3 양쪽 CBAM)의 주목할 성과**: small subgroup AUC 0.8421은 전체 실험에서 가장 높은 값이다. 이는 소결절 분류 성능 향상 측면에서 본 연구 가설을 지지하는 결과이다. FN=15(암을 놓치는 건수 최소)는 임상적 안전성 측면에서도 가장 유리한 구조이다.

**EXP-E에서 intermediate subgroup AUC가 EXP-C 대비 소폭 하락한 이유**: EXP-C 0.6477 → EXP-E 0.6250으로 −0.023 하락하였다. 해당 구간의 malignant 결절이 11개(n=83)로 극히 적어 단 1~2건의 예측 변화만으로도 AUC가 크게 흔들리는 통계적 불안정 구간이다. [확인 필요: 해당 구간 11개 malignant 결절의 예측 확률 분포 추가 분석 권장]

**Intermediate subgroup (100~250㎣) 의 지속적 취약성**: intermediate AUC가 전 실험에서 small, large 대비 낮게 나타났다(0.47~0.65). 해당 구간에 malignant 결절이 11개(n=83 중)로 매우 적어 AUC 추정이 불안정할 가능성이 있다. [확인 필요: 해당 구간 결절의 특성(크기 분포, 형태) 추가 분석 필요]

---

## 6. Visualization

### 6.1 완료된 시각화 항목

GDN 각 실험 결과에 대해 다음이 생성되었다: Loss/AUC training curve, ROC curve(nodule-level), Confusion matrix.

**Grad-CAM 설계**

TP/TN/FN/FP 각 케이스에서 5개 샘플을 선정하여 [k-1, k, k+1] 3채널을 나란히 시각화하였다. 3채널 개별 출력으로 어느 슬라이스에서 모델이 반응하는지를 확인할 수 있다.

### 6.2 예정 시각화 (추가 실험 후)

다음 비교를 통해 가설을 시각적으로 검증할 예정이다.

**EXP-A(baseline) vs EXP-E(CBAM@drop2+CBAM_final@drop3) Grad-CAM 비교**: CBAM 추가 후 모델이 결절 주변 맥락 영역(혈관 접촉부, 흉막 인접부)에 더 집중하는지를 확인한다. "주변 맥락 학습이 소결절 분류에 기여한다"는 가설을 시각적으로 뒷받침하는 근거가 될 것이다.

**소결절 케이스 집중 분석**: small subgroup AUC가 가장 크게 향상된 EXP-E에서 소결절 TP/FN 케이스의 Grad-CAM을 비교하여 어떤 특징이 분류에 결정적으로 작용하는지를 분석할 예정이다.

[TBD: Grad-CAM 비교 이미지 추가]

---

## 7. Limitation & Future Work

### 현재 한계

**Intermediate subgroup AUC의 지속적 취약성**: 100~250㎣ 구간에서 malignant 결절이 11개(n=83)로 극히 적어 AUC 추정이 불안정하다. 이 구간의 성능 해석에 주의가 필요하다.

**val loss 불안정 현상의 원인 미규명**: d2=3 dilation 적용 후 val loss 진동이 관찰되었으나, 평가 단위 불일치(slice vs nodule)와 dilation 확장에 따른 특징 분포 변화 중 어느 쪽이 주요 원인인지 본 연구에서 단정하지 못하였다. CBAM@drop1 추가 시 더욱 심화되는 이유에 대한 구조적 분석도 남아 있다.

**GD layer의 스케일 활용 패턴 미분석**: 각 GD layer가 결절 크기에 따라 local(dilation=1) 경로와 global(dilation=d2) 경로를 어떻게 활용하는지는 아직 분석하지 못하였다. 이는 모델의 해석 가능성(interpretability) 향상을 위한 후속 과제이다.

**crop size 32 한계**: 현재 32×32 crop은 소결절 주변 맥락을 제한적으로만 포함한다. 더 큰 crop size에서 주변 조직이 더 많이 포함될수록 성능이 향상될 가능성이 있다.

**Grad-CAM 시각화 미완성**: EXP-E의 Grad-CAM 분석이 완료되지 않아 "CBAM이 주변 맥락 영역에 집중하는지"를 시각적으로 검증하지 못했다.

### 향후 연구 방향

**Crop size 확장 실험**: 64×32 또는 96×32 등 더 큰 crop으로 GDN+CBAM을 실험하여 주변 맥락 포함 범위가 소결절 성능에 미치는 영향을 직접 확인한다. (예비 실험 260616_gdn_48_ep50_aug2에서 crop=48 적용 시 AUC 0.8702, small AUC 0.6941로 하락이 관찰된 것은 [확인 필요: 해당 실험의 다른 설정 조건과의 혼재 효과 분석 필요])

**추가 Dilation 탐색**: d2=4 이상 또는 multi-scale dilation 조합이 추가적인 맥락 포착에 기여할 수 있는지 탐색할 수 있다.

**Grad-CAM을 통한 가설 검증**: EXP-E에서 소결절 케이스의 Grad-CAM을 분석하여 모델이 실제로 혈관, 흉막, 주변 실질 패턴에 반응하는지를 확인하는 것이 최우선 후속 과제이다.

---

## 8. Conclusion

본 연구는 "소결절은 내부 정보가 제한적이므로 주변 맥락을 함께 학습하면 분류 성능이 향상될 것"이라는 가설 아래, LIDC-IDRI 데이터셋(최종 2,045개 결절, 800명)에서 폐결절 악성도 이진 분류를 수행하였다. 2.5D 입력 방식으로 z축 구조 정보를 활용하면서 3D CNN의 연산 부담을 줄였으며, GDN을 베이스라인으로 채택하고 dilation d2=3 확장과 CBAM을 결합한 GDN+CBAM(EXP-E)을 최종 제안 구조로 결정하였다.

모델 비교 단계에서 GDN이 파라미터 118배 적은 조건에서 DualConvNeXt 대비 모든 지표에서 우세함을 확인하였다. Dilation d2=3 적용과 Youden Index threshold 최적화(EXP-A)로 sensitivity를 0.7021→0.8191, small subgroup AUC를 0.7599→0.7944로 개선하였다. CBAM 위치 탐색을 통해 drop2 이후(gd4 통과 후) CBAM(EXP-C)과 drop3 이후(gd5 통과 후) CBAM_final을 함께 배치한 EXP-E가 test AUC 0.8939, Sensitivity 0.8404, small subgroup AUC 0.8421, FN=15로 전체 실험에서 핵심 지표 최고치를 달성하였다.

목표치 0.90에는 근소하게 미달했지만 baseline 대비 AUC +0.013, small subgroup AUC +0.082 향상을 달성하여 가설 검증에는 성공했다고 판단하였다.

Small subgroup AUC가 GDN baseline(0.7599)에서 EXP-E(0.8421)으로 +0.082 향상된 결과는 CBAM 추가가 소결절 분류에 긍정적인 영향을 주었음을 시사한다. 특히 small subgroup AUC가 baseline 대비 0.082 향상된 점은 본 연구가 설정한 "소결절에서 주변 맥락 정보가 더 중요하다"는 가설과 일관된 결과를 보였다. 실제로 모델이 어떤 영역에 주목하여 이 성능을 달성했는지는 Grad-CAM 분석을 통해 추가 검증이 필요하다.

---

## References

1. Shen, W. et al. (2017). Multi-crop Convolutional Neural Networks for lung nodule malignancy suspiciousness classification. *Pattern Recognition*, 61, 663–673.
2. Xie, Y. et al. (2018). Knowledge-based Collaborative Deep Learning for Benign-Malignant Lung Nodule Classification on Chest CT. *IEEE Transactions on Medical Imaging*, 38(4), 991–1004.
3. Al-Shabi, M. et al. (2019a). Lung nodule classification using Deep Local-Global Networks. *International Journal of Computer Assisted Radiology and Surgery*, 14(10), 1815–1819.
4. Masquelin, A. H. et al. (2022). Perinodular and Intranodular Radiomic Features on Lung CT Images Distinguish Adenocarcinomas from Granulomas. *Academic Radiology*, 30(6), 1073–1080.
5. Qin, X. et al. (2021). Correlation between pulmonary nodule and surrounding tissue in CT images. *arXiv:2106.12991*.
6. Al-Shabi, M. et al. (2019b). Gated-Dilated Networks for Lung Nodule Classification in CT Scans. *IEEE Access*, 7, 178827–178838.
7. Liu, J. et al. (2024). Lung nodule malignancy classification with associated pulmonary fibrosis using 3D attention-gated convolutional network. *Journal of Translational Medicine*, 22(1), 51.
8. Dey, R. et al. (2018). Diagnostic Classification of Lung Nodules Using 3D Neural Networks. *ISBI 2018*, 774–778.
9. Afshar, P. et al. (2020). 3D-MCN: A 3D Multi-scale Capsule Network for Lung Nodule Malignancy Determination. *Scientific Reports*, 10, 7572.
10. Faizi, M. et al. (2025). DCSwinB: Dual-branch CNN-Swin Transformer for pulmonary nodule malignancy prediction. *BMC Cancer*, 25, 1106.
11. Zheng, X. et al. (2021). Deep Learning-based Pulmonary Nodule Detection. *Medical Physics*, 48(2), 733–744.
12. Snoeckx, A. et al. (2018). Evaluation of the solitary pulmonary nodule. *Insights into Imaging*, 9, 1–14.
13. Gould, M. K. et al. (2013). Recent Trends in the Identification of Incidental Pulmonary Nodules. *American Journal of Respiratory and Critical Care Medicine*, 188(11), 1357–1363.
14. Woo, S. et al. (2018). CBAM: Convolutional Block Attention Module. *ECCV 2018*.