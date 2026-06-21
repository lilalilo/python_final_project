# 연구 코드 최적화 — ACE 기반 Scene Coordinate Regression 코드 리팩터링

----------------------------------------------------------------------------------------

본 저장소는 [ACE (Accelerated Coordinate Encoding, CVPR 2023)](https://nianticlabs.github.io/ace) 를
베이스라인으로 하는 카메라 재지역화(visual relocalization) 연구 코드와, 이를 대상으로 수행한 **코드 최적화 과제**의
개선 전/후 코드를 함께 담고 있다.

- **베이스라인:** ACE — CNN 인코더로 특징을 추출하고 픽셀별 3D 장면 좌표를 회귀(SCR)하여 6DoF 카메라 포즈를 추정.
- **본 연구의 변형:** ACE의 무작위 특징 샘플링 대신, **PIDNet 시맨틱 분할 + 컨텍스트 맵**으로부터 건물·엣지 등
  고유 구조 기반의 샘플링 마스크를 생성하도록 확장.
- **본 과제:** 위 변형 과정에서 누적된 구조적·연산적 비효율을 진단하고, 수업에서 학습한 개념(자료구조, 데코레이터,
  지연 평가, 클래스 설계)을 적용해 코드를 리팩터링한 뒤 **개선 전/후를 정량 비교**한다.

목차:

- [저장소 구조](#저장소-구조)
- [설치](#설치)
- [데이터셋](#데이터셋)
- [실행 방법](#실행-방법)
  - [학습](#학습)
  - [평가](#평가)
- [적용한 최적화](#적용한-최적화)
- [성능 측정 재현 (벤치마크)](#성능-측정-재현-벤치마크)
- [결과 요약](#결과-요약)
- [원본 ACE 참고](#원본-ace-참고)

## 저장소 구조

루트의 `*.py` 는 **최적화가 모두 적용된 최종 코드**(= `src/after` 와 동일)이며, 실행 시 사용된다.
개선 전/후 비교는 `src/` 아래 두 폴더로 보존되어 있다.

```text
final_renew/
├─ train_ace.py / test_ace.py        # 학습 / 평가 진입점
├─ ace_trainer.py                    # 학습 파이프라인 (최적화 D·B·C 적용)
├─ ace_network.py                    # 네트워크 정의 (최적화 A 적용)
├─ dataset.py / ace_loss.py / ...    # 그 외 ACE 구성요소
├─ src/
│  ├─ before/                        # 최적화 전 (baseline) 전체 .py
│  └─ after/                         # 최적화 후 (A+B+C+D) 전체 .py
├─ datasets/Cambridge_KingsCollege/  # 실험 데이터 (train/test)
├─ PIDNet/                           # 분할 백본 + 사전학습 가중치(camvid)
├─ output/                           # 학습된 head 가중치 및 로그
└─ report/                           # 보고서 PDF
```

> `src/before` 와 `src/after` 는 동일한 측정 계측 코드(`[MEM:*]` 로그)를 포함하되, 최적화 적용 여부만 다르다.
> 따라서 두 폴더를 루트로 교체하여 동일 조건에서 개선 전/후를 비교할 수 있다.
>
> 위는 전체 프로젝트 구조이며, `datasets/`·`PIDNet/`·`segformer/`·`output/`·가중치(`*.pt`) 등 대용량 자산은
> `.gitignore` 로 저장소에서 제외되어 있다(획득 방법은 [설치](#설치) 참고).

## 설치

본 코드는 ACE와 동일한 PyTorch 환경에서 동작하며, 추가로 분할 모듈에 필요한 의존성이 있다.
환경 및 RANSAC 빌드는 원본 저장소의 안내를 함께 참고한다.

- ACE: <https://github.com/nianticlabs/ace> (설치 방법 참고)
- DSAC*: <https://github.com/vislearn/dsacstar> (설치 방법 참고)

```shell
# 1) conda 환경 생성 (본 저장소의 environment.yml 사용)
conda env create -f environment.yml
conda activate acecontour   # 본 문서의 모든 명령은 이 환경에서 실행

# 2) 6DoF 포즈 추정을 위한 DSAC* RANSAC C++ 바인딩 빌드 (원본 ACE와 동일, 별도 설치)
#    dsacstar 소스는 원본 ACE/DSAC* 저장소에서 받아 빌드한다.
cd dsacstar
python setup.py install
cd ..
```

> `environment.yml` 은 `dsacstar` 를 포함하지 않는다. `dsacstar` 는 PyPI 패키지가 아니라 위와 같이
> 소스에서 직접 빌드·설치해야 하기 때문이다(원본 ACE도 동일하게 안내한다).

추가 의존성 / 별도 준비 자원 (용량 문제로 본 저장소에는 코드만 포함하며, 아래 가중치·데이터는 별도 배치):

- **`kornia`** — 샘플링 마스크 생성(`downsampled_maskbuilder.py`)의 형태학 연산에 사용 (environment.yml에 포함).
- **PIDNet 코드 및 사전학습 가중치** — [PIDNet 저장소](https://github.com/XuJiacong/PIDNet)에서 받아
  `PIDNet/` 에 배치한다. 코드는 camvid 가중치(`PIDNet/pretrained_models/camvid/PIDNet_S_Camvid_Test.pt`)를
  저장소 기준 상대 경로로 참조한다.
- **`ace_encoder_pretrained.pt`** — ACE 사전학습 인코더. 원본 ACE 저장소에서 받아 루트에 배치한다.

## 데이터셋

실험은 **Cambridge Landmarks — KingsCollege** 장면으로 수행하였다(저장소의 `datasets/Cambridge_KingsCollege`).
ACE의 데이터 포맷(DSAC* 호환)을 그대로 사용한다. 다른 장면이 필요하면 원본 ACE의 다운로드 스크립트를 사용한다.

```shell
cd datasets
./setup_cambridge.py        # datasets/Cambridge_{KingsCollege, ...}
```

## 실행 방법

### 학습

본 레포지토리에는 데이터셋 및 관련 모델이 포함되어 있지 않으므로, 설치 섹션을 참고하여 별도로 다운로드한다.

```shell
./train_ace.py <scene 경로> <출력 맵 파일>
# 예시:
./train_ace.py datasets/Cambridge_KingsCollege output/final.pt
```

학습이 끝나면 다음과 같이 버퍼 생성/학습/전체 시간이 출력된다.

```
Done without errors. Creating buffer time: 101.2 seconds. Training time: 301.7 seconds. Total time: 403.0 seconds.
```

### 평가

```shell
./test_ace.py <scene 경로> <출력 맵 파일>
# 예시:
./test_ace.py datasets/Cambridge_KingsCollege output/final.pt
```

평가는 임계값별 정확도(예: 10cm/5°)와 median 회전·병진 오차, 평균 처리 시간을 출력한다.
RANSAC(DSAC*)의 비결정성으로 인해 동일 모델이라도 결과에 미세 편차가 있으므로, 정확도는 여러 번 평가하여 평균을 사용한다.

## 적용한 최적화

| # | 항목 | 수업 개념 | 위치 | 핵심 변경 |
|---|---|---|---|---|
| A | 자료구조 | 해시맵·캐싱(메모이제이션) | `ace_network.py` (`CrossSimilarityShift`) | 매 forward 재생성되던 좌표 그리드를 `(H,W,device)` 키 dict로 캐싱 |
| D | 코드 구조 | 데코레이터·클로저·`functools.wraps` | `ace_trainer.py` | 분산된 `time.time()` 측정 6곳을 `@cuda_timeit` 1개로 통합(+CUDA `synchronize`) |
| B | 데이터 적재 | 즉시 vs 지연 평가 | `ace_trainer.py` (`create_training_buffer`) | 전체 N행 materialize 후 일부만 사용 → 샘플 인덱스 먼저 뽑아 선택 행만 gather |
| C | 구조/재현성 | `dataclass`·불변성·단일 책임 원칙 | `ace_trainer.py` (`TrainerACE`) | 설정을 `@dataclass(frozen=True) RunConfig`, 런타임 상태를 `TrainerState`로 분리 |

## 성능 측정 재현 (벤치마크)

별도의 벤치마크 스크립트 대신, 학습 파이프라인에 가벼운 계측 코드를 삽입하여 학습 실행 중 직접 측정한다.
`create_training_buffer` 는 다음 두 줄을 출력한다.

- `[MEM:batch_data] per-iter intermediate = X MB (N rows)` — 반복당 생성되는 중간 객체 크기
- `[MEM:buffer_fill] baseline=.. peak=.. intermediate=X MB` — 버퍼 채우는 구간의 peak 메모리(영구 버퍼 제외분)

**개선 전/후 비교 절차** (루트에 해당 버전을 교체하여 동일 조건에서 측정):

```shell
# (개선 후) 루트를 after 로 두고 측정
cp src/after/*.py .
python train_ace.py datasets/Cambridge_KingsCollege output/after.pt 2>&1 | tee after.log

# (개선 전) 루트를 before(baseline) 로 교체하여 측정
cp src/before/*.py .
python train_ace.py datasets/Cambridge_KingsCollege output/before.pt 2>&1 | tee before.log

# 측정 끝나면 루트를 after 로 복구
cp src/after/*.py .

# 값 추출
grep -E "MEM:batch_data|MEM:buffer_fill|Done without errors" before.log after.log
```

> 메모리 측정은 결정적이므로 1회면 충분하다. 학습 시간은 run 간 변동이 있으므로, 시간 비교가 필요하면 각 버전을
> 여러 번 실행해 평균과 표준편차를 함께 보고한다.

## 결과 요약

KingsCollege 장면, RTX 3060(12GB) 기준. 정확도는 평가 평균, 학습 시간은 개선 전 3회 평균.

| 지표 | 개선 전 (baseline) | 개선 후 (A+B+C+D) |
|---|---|---|
| Total time (s) | 399.7 ± 2.1 | 403.0 |
| 10cm/5° · 5cm/5° | 13.4% · 1.5% | 14.9% · 2.6% |
| Median Error (°/cm) | 0.3 / 22.4 | 0.4 / 23.0 |
| 반복당 중간 객체 | 143.9 MB | **8.2 MB (−94.3%)** |
| buffer-fill peak | 1516.2 MB | 1503.5 MB (−0.8%) |
| 설정 접근 구조 | argparse Namespace 69곳(가변) | `RunConfig` 단일 불변 객체 |
| 측정 코드 | `time.time()` 6곳 산재 | 데코레이터 1개 |

- **정확도:** 전 항목에서 개선 전과 동등(노이즈 범위 내) → 모든 변경이 동작을 보존.
- **학습 시간:** 개선 전후가 표준편차 범위 내에서 구분되지 않음(대상 연산이 병목이 아님).
- **메모리:** 반복당 중간 객체 −94.3%, 단 peak 메모리는 forward 활성화가 지배하여 거의 불변.
- **구조:** 측정·설정 관심사 분리로 가독성·재현성 향상.

자세한 분석과 한계는 `report/` 의 보고서를 참고.

## 원본 ACE 참고

본 코드는 ACE 및 DSAC* 파이프라인 위에 구축되었다. ACE를 사용하는 경우 다음을 인용:

```bibtex
@inproceedings{brachmann2023ace,
    title={Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses},
    author={Brachmann, Eric and Cavallari, Tommaso and Prisacariu, Victor Adrian},
    booktitle={CVPR},
    year={2023},
}
```

원본 프로젝트: <https://nianticlabs.github.io/ace> · [Arxiv](https://arxiv.org/abs/2305.14059)

License: 원본 ACE 코드 부분은 Copyright © Niantic, Inc. 2023 (Patent Pending). 해당 `LICENSE` 조건을 따른다.
