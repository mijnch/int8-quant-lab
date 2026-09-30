# 작업 노트 — 막힌 곳, 알아낸 것, 근거

결과 수치는 [results/results.md](../results/results.md)에 있다. 여기에는 그 수치에 이르기까지
확인한 사실과 **각 사실의 근거(원문 파일 또는 실측)** 를 적는다. 원문은 모두 이 작업에서 쓴
버전(LiteRT v2.2.0, TensorFlow v2.21.0, 그리고 두 저장소가 고정한 ruy·XNNPACK·NEON_2_SSE 커밋)에서
직접 읽었다.

## 1. 환경과 데이터

| 문제 | 확인한 것 | 대응 |
|---|---|---|
| tfds 기본 주소 `download.tensorflow.org`가 작업 환경의 프록시에서 403 | 같은 파일이 `storage.googleapis.com/download.tensorflow.org/…`에 있음 | 주소만 바꿔 받고, sha256을 tfds의 `checksums.tsv` 등록값과 대조 — 두 파일 모두 일치 |
| Python 3.13에서 tfds가 wav를 못 읽음 | tfds가 쓰는 pydub이 `audioop`을 import하는데 3.13에서 표준 라이브러리에서 빠짐 | `audioop-lts` 설치 |
| TF 2.21에서 레퍼런스 SavedModel이 안 열림 | 기본 Keras 3는 옛 SavedModel을 `load_model`로 열지 못하고, `tf.saved_model.load`는 옵티마이저 슬롯 복원에서 실패 | `tf-keras` 2.21 + `TF_USE_LEGACY_KERAS=1` — 레퍼런스 quantize.py와 같은 `from_keras_model` 경로 |
| 레퍼런스 전처리를 그대로 옮겼는가 | MLPerf `get_dataset.get_training_data`를 TF 2.21에서 직접 돌려 같은 분할의 특징을 만들고 비교 | 테스트 4,890 · 검증 10,102 · 보정 120개 모두 **비트 일치**(`results/data_check.json`) |

## 2. MLPerf 배포 파일에 대해 알게 된 것

- **`kws_ref_model_float32.tflite`는 순수 float32가 아니다.** 합성곱 5개 층의 가중치가 int8이다.
  레퍼런스 `quantize.py`가 `converter.optimizations = [Optimize.DEFAULT]`를 설정한 **뒤에**
  이 파일을 변환하기 때문에 동적 범위 양자화가 걸렸다(깊이별·FC 가중치는 원소 수가 적어 float로 남음).
  그래서 FP32 기준선은 SavedModel을 최적화 없이 다시 변환해 따로 만들었다.
- **배포된 `.tflite` 두 개와 SavedModel은 서로 다른 학습 결과로 보인다.** 층별 가중치의 상관계수:
  배포 int8 ↔ 배포 "float32" 10개 층 모두 0.9998 이상, SavedModel(BN 접은 뒤) ↔ 배포 int8은
  10개 층 모두 −0.07~0.07(`results/t1.json`의 `weight_corr_savedmodel_vs_mlperf_int8`).
  같은 SavedModel을 TF 2.21로 변환하면 int8 가중치가 층마다 대부분 다르다(`results/t1.json`의
  `vs_mlperf_int8`). 원인(어느 쪽이 나중에 교체됐는지)은 확인하지 못했다. 그래서 T1(파라미터 재현)은
  SavedModel을 직접 변환한 결과로 하고, 커널 대조(T2·T3)는 두 int8 모델 모두로 했다.
- **레퍼런스 평가 코드의 입력 변환은 반올림도, 범위 자르기도 하지 않는다.**
  `eval_quantized_model.py`: `np.array(dat/input_scale + input_zero_point, dtype=np.int8)`.
  float→int8 변환은 0 쪽으로 버리고, int8 범위 밖 값(테스트 입력의 약 0.13%)은 감긴다.
  반올림+자르기로 바꾸면 정확도가 0.18~0.29%p 오르지만 McNemar p=0.20~0.43로 우연과 구별되지 않는다.

## 3. T1 — 변환기의 규칙 (TensorFlow v2.21.0 원문)

| 대상 | 규칙 | 원문 |
|---|---|---|
| 보정 통계 | 샘플마다 텐서 min/max를 누적 | `lite/tools/optimize/calibration/calibration_logger.h` (MinMax) |
| 활성값 범위 | min ≤ 0 ≤ max가 되도록 넓힘, 폭이 1e-6 미만이면 ±1e-6 | `quantization_utils.h` ConvertStatsToQDQs, TensorRangeSanityCheck |
| 활성값 scale·zp | 파이썬 경로는 `legacy_float_scale=true`로 고정 → DownCastScale: scale = float32(max−min)/255 (float32 연산), zp = round(float32(−128 − min/scale)) | `compiler/mlir/lite/python/converter_python_api.cc`, `quantization_lib/quantization_utils.cc` DownCastScale |
| 가중치 scale | 채널별 max\|w\|, 좁은 범위 [−127, 127] → float32(2·max)/254 | ExtractMinMaxFromAttr + DownCastScale |
| 가중치 값 | **나눗셈이 아니라 float32 역수 곱**: inv = float32(1.0/scale), q = round(float32(w·inv)) | `QuantizeLegacy` → `compiler/mlir/tools/optimize/quantization_utils.cc` SymmetricPerChannelQuantizeValues |
| 텐서별 가중치 값 | inv = float32(127/max\|w\|) | `toco_legacy/portable_tensor_utils.cc` PortableSymmetricQuantizeFloats |
| bias scale | float32(double(s_in)·double(s_w)) | GetUniformQuantizedTypeForBias |
| bias 값 | inv = float32(1.0/scale), q = round(float32(b·inv)), ±(2³¹−1)로 자름 | SymmetricBiasQuantize |
| AVERAGE_POOL·RESHAPE 출력 | 입력과 같은 scale·zp (자기 보정 통계를 쓰지 않음) | 실측 |
| SOFTMAX 출력 | scale 1/256, zp −128 고정 | `activations.cc` SoftmaxPrepare가 이 값을 요구 |

처음에는 일반 MLIR 경로(`UniformSupport.h`, double 나눗셈)로 구현했는데 bias 588개 중 1개가 1 차이
났다(op 7 채널 62: b/scale = 8284.4994 → 8284, 변환기는 8285). float32로 역수를 곱하면 정확히 8284.5가
되어 0에서 먼 쪽으로 8285가 된다. 이 차이를 따라가 `QuantizeLegacy` 경로를 찾았다.

보정 min/max를 직접 다시 재면, **TensorFlow에 내장된 TFLite 인터프리터(최적화 커널, 위임 없음)**
로는 14개 모두 비트 일치하고, LiteRT 2.2.0으로는 12개가 일치한다(FC 출력 최댓값이 1 ulp 다르고
그 뒤 softmax에 번진다). 변환기의 보정기(`CalibrationWrapper`)가 TF 쪽 바이너리를 쓰기 때문이다.

## 4. T2·T3 — 런타임마다 다른 재양자화

같은 `.tflite`라도 커널 경로마다 재양자화 산술이 다르다. 모두 "실수 배율을 곱하고 반올림"하지만
**동률(정확히 .5)을 어느 쪽으로 보내는지, 몇 번 반올림하는지**가 다르다.

| 이름 | 계산 | 쓰는 곳(x86-64, 원문) |
|---|---|---|
| double | SRDHM(동률 +∞) → RoundingDivideByPOT(동률 0에서 먼 쪽) | 참조 커널의 CONV·DEPTHWISE, 최적화 DEPTHWISE의 나머지 채널 (`common.cc`) |
| float | round(double(acc) × double 배율) | 참조 커널의 FULLY_CONNECTED (`reference/integer_ops/fully_connected.h`) — LiteRT 2.x에서 바뀐 부분 |
| ruy | ((x·M + 2³⁰) >> 31), 이어서 (v + 2^(r−1)) >> r — 둘 다 동률 +∞ | 최적화 커널의 CONV·FC (`cpu_backend_gemm.h`: x86 기본값은 int8 → ruy; ruy `kernel_avx512.cc`) |
| neon8 | 앞 8·⌊C/8⌋개 채널: vqrdmulhq → vrshlq(동률 +∞), 나머지 채널: double | 최적화 커널의 DEPTHWISE (`optimized_ops.h` Quantize; x86에서는 NEON_2_SSE가 NEON 명령을 SSE로 흉내) |
| fp32 | rint(float32(acc) × float32(s_in·s_w/s_out)) — 동률은 짝수 쪽 | XNNPACK의 CONV·DEPTHWISE·FC (`convolution-nhwc.c`, `qs8-qc8w-gemm/…fp32…`) |
| single | 정확한 곱을 한 번 반올림(동률 +∞) | `TFLITE_SINGLE_ROUNDING` 빌드 — 이 휠에서는 쓰이지 않음을 실측으로 확인 |

- AVERAGE_POOL_2D와 SOFTMAX(int8)는 XNNPACK에 위임되지 않아 TFLite 커널이 돈다(실행 계획으로 확인).
- int8 SOFTMAX는 참조 커널이 gemmlowp 고정소수점 exp·역수(`reference/softmax.h`), 최적화 커널이
  float32 exp 표 256칸(`optimized_ops.h`)을 쓴다. 표는 C 라이브러리의 `expf`로 만들므로 numpy 구현도
  ctypes로 같은 `expf`를 부른다.
- **이 차이를 찾은 방법.** KWS 모델만으로는 대부분 드러나지 않는다: 모든 합성곱에 ReLU가 붙어 있어
  음수 결과가 −128로 잘리고, 규칙 차이는 음수의 동률에서만 나기 때문이다. 그래서 scale·zero-point·
  활성함수·보폭·패딩·채널 수를 무작위로 바꾼 **단일 연산 모델을 flatbuffer로 직접 만들어**
  (`src/qlab/synth.py`) 세 런타임에 돌렸다. 처음 정책(최적화 DEPTHWISE = double)은 100개 중 97개만
  맞았고, 틀린 3개가 모두 채널 8개짜리라는 데서 `optimized_ops::Quantize`의 8채널 묶음 경로를 찾았다.
- XNNPACK은 연산을 묶어 한 덩어리로 위임하므로 중간 텐서가 보이지 않는다.
  `experimental_disable_delegate_node_fusion=True`로 연산마다 따로 위임시키면 층별 출력을 비교할 수 있다.

## 5. 구현상 주의

- 누산은 float64 행렬곱으로 계산한다. |x−zp| ≤ 255, |w| ≤ 127, 항 K개 → |acc| ≤ 255·127·K가 2⁵³보다
  작으면 float64 합이 정확한 정수라서, 실행 전에 이 한계를 확인한다(`kernels._exact_matmul`).
- 깊이별 합성곱을 패치 배열로 펼치면 테스트 세트 전체에서 메모리가 6 GB를 넘었다. 필터 칸마다 입력을
  밀어 더하는 방식으로 바꿨다(결과 동일, 무작위 모델 검증 재통과).
- C++ 정수 나눗셈은 0 쪽으로 버리고 numpy `//`는 내림이다. `std::round`는 동률을 0에서 먼 쪽으로,
  `np.round`는 짝수 쪽으로 보낸다. 둘 다 별도 함수로 구현했다(`fixedpoint.trunc_div`, `round_half_away`).

## 6. 확인하지 못한 것

- **다른 플랫폼.** 위 표는 Linux x86-64(AVX-512)에서 확인했다. ARM(NEON 원본, 점곱 3×3 커널)이나 다른
  컴파일 설정에서는 경로가 달라질 수 있다. CI는 Ubuntu와 Windows에서 같은 테스트를 돌린다.
- **마이크로컨트롤러.** TFLite Micro와 CMSIS-NN 커널의 재양자화는 확인하지 않았다. 하드웨어가 없어
  지연·전력도 재지 않았다.
- **MLPerf 배포 파일의 출처.** SavedModel과 `.tflite`가 왜 다른 모델인지는 모른다.
