# int8-quant-lab — INT8 양자화, 라이브러리 출력과 비트 단위로 맞춰 보기

[![tests](https://github.com/mijnch/int8-quant-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/mijnch/int8-quant-lab/actions/workflows/ci.yml)

MLPerf Tiny의 키워드 인식(KWS) 레퍼런스 모델(DS-CNN)을 대상으로, TensorFlow Lite(LiteRT)의
**INT8 양자화 규칙과 정수 추론 산술을 numpy로 다시 구현해 라이브러리 출력과 비트 단위로 대조**하고,
양자화 설계 변수가 정확도에 주는 영향을 쟀습니다.

> 한 줄 요약 — 변환기가 정한 scale·zero-point·int8 값(가중치 22,016개, bias 588개)을 전부 똑같이
> 재현했고, 세 가지 실행 경로(참조 커널·최적화 커널·XNNPACK)의 출력을 테스트 세트 전체에서
> 한 비트도 틀리지 않게 재현했습니다. 그 과정에서 **같은 모델이라도 실행 경로와 플랫폼에 따라
> 반올림 규칙이 6가지로 갈린다**는 것을 원문과 실측으로 확인했습니다.

## 핵심 결과

| | 결과 | 근거 |
|---|---|---|
| T1 양자화 파라미터 재현 | 활성값 scale·zp **14/14**, int8 가중치 **22,016/22,016**, int32 bias **588/588** 일치 | [results.md §2](results/results.md#2-t1--변환기가-정한-양자화-파라미터값의-재현) |
| T2·T3 정수 추론 재현 | 테스트 4,890개 × 출력 12개, 모델 2개 × 런타임 3개 **전부 불일치 0** · 무작위 단일 연산 모델 500개 × 런타임 3개 전부 일치 (Linux 측정, Windows는 CI로 확인) | [§3](results/results.md#3-t2t3--numpy-정수-커널-대-litert-세-실행-경로-비트-단위) |
| FP32 → INT8 | 정확도 92.17% → 92.33%, 파일 99.4 KB → 48.5 KB. 차이는 **우연과 구별되지 않음**(McNemar p=0.40) | [§4](results/results.md#4-fp32-대-int8--정확도크기) |
| 4비트 가중치(시뮬레이션) | 채널별 91.1% 대 **텐서별 대칭 57.4%** · 텐서별 비대칭 80.2%. 1×1 합성곱 한 층의 영향이 가장 큼 | [§5 C·D](results/results.md#5-설계-변수-실험) |

8비트에서는 채널별/텐서별, 대칭/비대칭, 보정 샘플 10개 이상 여부 모두 정확도 차이가 통계적으로
구별되지 않았습니다. 차이는 보정 샘플 1개(평균 91.2%)와 4비트에서 드러났습니다.

## 무엇을 했나

**T1 — 파라미터 재현** (`src/qlab/spec.py`, `t1.py`). 보정 데이터에서 잰 min/max로 활성값
scale·zero-point를, float 가중치로 채널별 scale과 int8 값을, bias를 int32로 계산해 TF 2.21
변환기의 결과와 비교합니다. 처음에는 bias 1개가 1 차이 났는데, 파이썬 변환 경로가 나눗셈이 아니라
**float32 역수를 곱하는 옛 코드 경로(`QuantizeLegacy`)** 를 탄다는 것을 원문에서 찾아 맞췄습니다.

**T2·T3 — 정수 산술 재현** (`fixedpoint.py`, `kernels.py`, `model.py`). CONV·DEPTHWISE·FC·
AVERAGE_POOL·SOFTMAX를 int32 누산 → 재양자화 → zero-point → 포화의 정수 연산으로 구현했습니다.
재양자화는 실행 경로와 플랫폼마다 달라서, 원문을 읽고 6가지를 구현했습니다.

| 이름 | 계산 | Linux x86-64 | Windows x86-64 |
|---|---|---|---|
| double | SRDHM(동률 +∞) → RoundingDivideByPOT(동률 0에서 먼 쪽) | 참조 CONV·DEPTHWISE | 참조 CONV·DEPTHWISE, 최적화 DEPTHWISE |
| float | round(double(누산) × double 배율) | 참조 FC | 참조 FC |
| ruy | 두 번 반올림, 둘 다 동률 +∞ | 최적화 CONV·FC (ruy SIMD 커널) | — |
| neon8 | 8채널 묶음은 NEON 경로(x86에선 SSE로 흉내), 나머지 채널은 double | 최적화 DEPTHWISE | — |
| single | 정확한 곱을 한 번 반올림, 동률 +∞ | — | 최적화 CONV·FC (ruy 표준 C++ 경로) |
| fp32 | rint(float32(누산) × float32 배율), 동률은 짝수 쪽 | XNNPACK CONV·DEPTHWISE·FC | XNNPACK CONV·DEPTHWISE·FC |

Windows 휠(MSVC 빌드)에서는 ruy의 SIMD 커널과 x86 NEON→SSE 경로가 GCC/Clang용 컴파일 조건
때문에 빠져 있어 최적화 커널의 규칙이 다릅니다. Windows CI에서 처음 드러났고, 원문 조건으로 확인한 뒤
`scripts/probe_rounding.py`(연산·경로별로 맞는 규칙을 표로 출력)를 CI에 넣어 확인합니다.

실제 모델에서는 모든 합성곱 뒤의 ReLU가 음수 결과를 잘라내 이 차이가 대부분 가려집니다. 그래서
scale·zero-point·활성함수·보폭·채널 수를 무작위로 바꾼 **단일 연산 `.tflite`를 직접 만들어**
(`synth.py`) 세 런타임과 대조했고, 여기서 최적화 깊이별 합성곱의 8채널 묶음 경로까지 찾아냈습니다.
발견 과정과 원문 위치는 [docs/notes.md](docs/notes.md)에 적었습니다.

**비교 실험** (`scripts/experiments.py`). 채널별 대 텐서별(실제 int8 변환), 보정 샘플 1·10·100·500개
(시드 5개), 가중치 대칭 대 비대칭 × 8·4비트(시뮬레이션), 4비트 층별 민감도. 모든 정확도 비교에
McNemar 정확 검정을 붙였고, 비교가 많아 p < 0.002(본페로니)만 확실한 차이로 읽습니다.

## 측정한 것 / 측정하지 않은 것

| 측정함 | 측정하지 않음 |
|---|---|
| top-1 정확도 (tfds 테스트 분할 **4,890개 전체**) | **MLPerf Tiny 공식 제출이 아닙니다.** 공식 평가는 1,000개 부분집합을 기기 위에서 잽니다. 같은 과제·지표로 자체 측정한 값입니다 |
| `.tflite` 크기, 파라미터 수, MAC | 마이크로컨트롤러 지연·전력 — 하드웨어가 없습니다 |
| numpy 재현 대 라이브러리 불일치 개수 | TFLite Micro·CMSIS-NN 커널, ARM 플랫폼의 커널 경로 |
| x86 지연시간 (클라우드 VM, 스레드 1개 — **MCU를 대표하지 않음**) | 모델 학습 — MLPerf 레퍼런스 모델을 그대로 썼습니다 |

## 알게 된 것 (MLPerf 배포 파일)

- `kws_ref_model_float32.tflite`는 이름과 달리 **합성곱 가중치가 int8인 동적 범위 양자화 모델**입니다.
  레퍼런스 `quantize.py`가 `Optimize.DEFAULT`를 켠 뒤 변환하기 때문입니다. FP32 기준은 따로 만들었습니다.
- 배포된 `.tflite` 두 개와 SavedModel은 **서로 다른 학습 결과로 보입니다** — 층별 가중치 상관계수가
  10개 층 모두 −0.07~0.07입니다(배포 `.tflite` 두 개끼리는 10개 층 모두 0.9998 이상).
- 레퍼런스 평가 코드는 입력을 int8로 바꿀 때 **반올림도 범위 자르기도 하지 않습니다**
  (`np.array(x/scale + zp, dtype=np.int8)`) — 입력값의 0.13%가 int8 범위를 넘어 감깁니다.

## 재현 방법

테스트만(TensorFlow·데이터 불필요, Python 3.10~3.14):

```bash
pip install -r requirements.txt
pytest                      # 29개: 고정소수점 단위 검증, T1 재현, 세 런타임 비트 일치
```

전체 실험(Python 3.10~3.13, 약 2.5 GB를 내려받고 tfds 변환 후 디스크 약 11 GB 사용):

```bash
pip install -r requirements-experiments.txt
git clone https://github.com/mlcommons/tiny   # 사용한 커밋: 4addd0f
SM=tiny/benchmark/training/keyword_spotting/trained_models/kws_ref_model
python scripts/prepare_data.py --tfds-dir ~/tfds --mlperf-dir tiny   # MFCC + 레퍼런스 전처리와 대조
python scripts/verify_t1.py --saved-model $SM      # T1 (models/kws_tf221_*.tflite를 다시 만듦 — 바이트 동일)
python scripts/verify_bitexact.py                  # T2·T3
python scripts/evaluate.py --saved-model $SM       # 정확도·크기·지연
python scripts/experiments.py --saved-model $SM    # 설계 변수 실험 (변환 약 30회)
python scripts/report.py                           # results/results.md, 그림
```

## 구조

```
src/qlab/
  fixedpoint.py  고정소수점 기본 연산, 재양자화 5종, gemmlowp exp·역수
  spec.py        T1: scale·zero-point, 가중치·bias 양자화 (TF 2.21 규칙)
  kernels.py     정수 커널: conv · depthwise · FC · average pool · softmax(2종)
  model.py       .tflite 직접 읽기(flatbuffer) · 망 전체 실행 · 런타임별 재양자화 정책
  runtime.py     LiteRT 세 실행 경로
  synth.py       검증용 단일 연산 모델 생성
  t1.py          T1 대조
  simulate.py    가중치 fake quantization
  stats.py       McNemar 정확 검정
  data.py        Speech Commands → MFCC          (TensorFlow 필요)
  convert.py     Keras → .tflite                 (TensorFlow 필요)
scripts/         prepare_data · verify_t1 · verify_bitexact · evaluate · experiments · report
                 probe_rounding (이 컴퓨터의 런타임이 어떤 반올림 규칙을 쓰는지 표로 — TF 불필요)
tests/           pytest (TensorFlow 없이)
models/          MLPerf 배포 모델·보정 인덱스, TF 2.21 변환 결과(int8·보정용·float32)
results/         results.md(자동 생성), *.json, figures/
docs/notes.md    막힌 곳과 알아낸 것, 원문 위치
```

## 만든 방식

계획(주제 선정, 재현 범위 T1~T3, 측정 지표, 정직성 원칙)은 제가 세웠습니다.
**구현·검증·측정·문서 작성은 제 요청에 따라 AI 코딩 도구(Claude Code)가 수행했습니다** — 이 저장소의
코드는 제가 직접 짠 것이 아닙니다. 대신 모든 주장은 원문 파일 위치(docs/notes.md)나 다시 돌릴 수 있는
측정(results/*.json, 테스트)으로 확인할 수 있게 했습니다.

pdf-ocr-korean-textbook에서는 ONNX Runtime의 동적 int8 양자화(`quantize_dynamic`, 가중치만 int8)를
썼습니다. 이 저장소는 그보다 한 단계 아래, 활성값까지 int8인 정적 양자화의 산술을 끝까지 따라갑니다.

## 한계와 다음 단계

- 정확도·지연 측정과 망 전체 대조는 Linux x86-64(AVX-512)에서 했습니다. Windows x86-64는 CI(규칙 탐침 +
  테스트)로만 확인했습니다. ARM(NEON 원본, 점곱 3×3 커널 등)과 macOS는 확인하지 않았습니다.
- 다음: 마이크로컨트롤러 실측(TFLite Micro, CMSIS-NN의 재양자화 대조), 양자화 인식 학습(QAT),
  층별 민감도에 따른 혼합 정밀도.

## 출처·라이선스

- 코드: MIT (LICENSE). 모델·보정 인덱스: MLPerf Tiny, Apache-2.0. 데이터: Speech Commands v2
  (Warden 2018, arXiv:1804.03209), CC BY 4.0 — 저장소에 넣지 않음. 자세한 내용은 [NOTICE](NOTICE).
- 규칙 원문: LiteRT v2.2.0, TensorFlow v2.21.0, gemmlowp, ruy 3286a34, XNNPACK 25b42df, NEON_2_SSE a15b489.

<details>
<summary>English summary</summary>

**int8-quant-lab** re-implements TensorFlow Lite / LiteRT INT8 quantization in NumPy and checks it
bit-for-bit against the library, using the MLPerf Tiny keyword-spotting reference model (DS-CNN,
Speech Commands v2).

- **T1 (parameters):** activation scales/zero-points (14/14), per-channel weight scales and all
  22,016 int8 weights, and all 588 int32 biases produced by the TF 2.21 converter are reproduced exactly.
  Matching the last bias required finding that the Python conversion path uses a legacy
  float32-reciprocal-multiply rule rather than division.
- **T2/T3 (integer inference):** conv, depthwise conv, fully connected, average pool and softmax
  reproduce the library output with zero mismatches on the full 4,890-clip test set, for two models
  and three execution paths (reference kernels, optimized kernels, XNNPACK), and on 500 randomly
  generated single-op models run on each of the three paths (measured on Linux; Windows checked in CI).
  Doing so required six requantization rules — the same model rounds differently depending on the
  kernel path and on the platform's build (the Windows wheel lacks ruy's SIMD kernels and the x86
  NEON-to-SSE path) — each traced to its source file.
- **Experiments:** FP32 92.17% vs INT8 92.33% (not distinguishable, McNemar p=0.40) at half the file size;
  at 8 bits, per-channel vs per-tensor, symmetric vs asymmetric and ≥10 calibration samples make no
  detectable difference, while 4-bit per-tensor symmetric weights collapse to 57.4% (simulation),
  with a single 1×1 conv layer contributing the most.
- **Findings about the MLPerf artifacts:** the file named `*_float32.tflite` is actually dynamic-range
  quantized; the distributed `.tflite` files and the SavedModel appear to come from different training runs.

Not an official MLPerf submission; no microcontroller latency/power was measured. The plan (scope,
metrics, honesty rules) is the author's; implementation, verification, measurement and documentation
were done by an AI coding assistant (Claude Code) at the author's request.

</details>
