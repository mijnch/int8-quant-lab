"""T1 — 양자화 파라미터(scale·zero-point)와 값 양자화를 TF 2.21 변환기와 같은 규칙으로 계산한다.

    real = (q - zero_point) × scale

규칙의 원문(TensorFlow v2.21.0):
- 활성값: `quantization_utils.h`(ConvertStatsToQDQs) → `FakeQuantSupport.cc`(fakeQuantAttrsToType)
  → `quantization_utils.cc`(DownCastScale). 파이썬 변환 경로는 `legacy_float_scale=true`로
  고정돼 있어(`converter_python_api.cc`), 최종 scale은 float32 연산으로 다시 계산된다.
- 가중치: `ExtractMinMaxFromAttr`(0 포함, 대칭) → narrow range [-127, 127] → DownCastScale.
- bias: scale = float32(입력 scale × 가중치 scale).
- 값 변환(상수): 파이썬 경로는 `QuantizeLegacy`(`quantization_lib/quantization_utils.cc`)를 타서
  나눗셈이 아니라 **float32 역수를 곱한 뒤** 반올림한다(`tools/optimize/quantization_utils.cc`
  SymmetricPerChannelQuantizeValues·SymmetricBiasQuantize, `portable_tensor_utils.cc`).
  일반 MLIR 경로(`UniformSupport.h`, double 나눗셈)는 비교용으로 `quantize()`에 남겼다 —
  두 규칙은 거의 같지만 드물게 1 차이 난다(results/t1.json).
"""
import numpy as np

from .fixedpoint import round_half_away

NEAR_ZERO = 1.0e-6          # kNearZeroTolerance


def _f32(x):
    return np.float32(x)


def activation_qparams(rmin, rmax, qmin=-128, qmax=127):
    """보정 데이터에서 잰 min/max → (scale: float32, zero_point: int).

    1) 범위가 0을 포함하도록 넓힌다  2) 너무 좁으면 ±1e-6 넓힌다
    3) scale = float32(max − min) / 255  (float32 연산)
    4) zero_point = round(float32(qmin − min / scale)), [qmin, qmax]로 자름
    """
    rmin = min(float(rmin), 0.0)
    rmax = max(float(rmax), 0.0)
    if abs(rmax - rmin) < NEAR_ZERO:
        rmin -= NEAR_ZERO
        rmax += NEAR_ZERO
    scale = (_f32(rmax) - _f32(rmin)) / _f32(qmax - qmin)
    zp_from_min = _f32(qmin - rmin / float(scale))
    if zp_from_min < qmin:
        zp = qmin
    elif zp_from_min > qmax:
        zp = qmax
    else:
        zp = int(round_half_away(zp_from_min))
    return np.float32(scale), zp


def weight_scales(w, axis):
    """가중치 대칭 양자화 scale. axis=None이면 텐서 하나에 scale 하나(per-tensor).

    scale_c = float32(2·max|w_c|) / 254   (= max|w_c| / 127, 좁은 범위 [-127, 127])
    """
    w = np.asarray(w, dtype=np.float32)
    if axis is None:
        absmax = np.abs(w).max(keepdims=True).reshape(1)
    else:
        red = tuple(i for i in range(w.ndim) if i != axis)
        absmax = np.abs(w).max(axis=red)
    if np.any(absmax == 0):
        raise ValueError("모든 값이 0인 채널은 이 구현이 다루지 않는다")
    two_max = absmax.astype(np.float32) - (-absmax.astype(np.float32))
    return (two_max / np.float32(254)).astype(np.float32)


def quantize(x, scale, zero_point, qmin, qmax, axis=None):
    """값 변환: clamp(round_half_away(x / scale + zp)). scale·zp는 axis 방향으로 브로드캐스트."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    scale = np.asarray(scale, dtype=np.float32).astype(np.float64)
    zero_point = np.asarray(zero_point, dtype=np.float64)
    if axis is not None and scale.size > 1:
        shape = [1] * x.ndim
        shape[axis] = -1
        scale = scale.reshape(shape)
        zero_point = np.broadcast_to(zero_point, scale.shape) if zero_point.size == 1 \
            else zero_point.reshape(shape)
    q = round_half_away(x / scale + zero_point)
    return np.clip(q, qmin, qmax)


def quantize_weights(w, axis, rule="legacy"):
    """→ (int8 가중치, float32 scale 배열). zero-point는 항상 0, 값은 [-127, 127].

    rule="legacy"(파이썬 변환기):
      per-channel  inv_c = float32(1.0 / double(scale_c)),  q = round(float32(w × inv_c))
      per-tensor   inv   = float32(127 / max|w|),            q = round(float32(w × inv))
    rule="divide"(일반 MLIR 경로): q = round(double(w) / double(scale))
    """
    s = weight_scales(w, axis)
    if rule == "divide":
        q = quantize(w, s, 0, -127, 127, axis=axis if s.size > 1 else None)
        return q.astype(np.int8), s
    w32 = np.asarray(w, np.float32)
    if axis is None:
        inv = np.float32(np.float32(127) / np.float32(np.abs(w32).max()))
        prod = w32 * inv
    else:
        inv = (1.0 / s.astype(np.float64)).astype(np.float32)
        shape = [1] * w32.ndim
        shape[axis] = -1
        prod = w32 * inv.reshape(shape)
    q = np.clip(round_half_away(prod.astype(np.float32)), -127, 127)
    return q.astype(np.int8), s


def bias_scales(input_scale, w_scales):
    """bias scale = float32(double(입력 scale) × double(가중치 scale))."""
    prod = np.float64(np.float32(input_scale)) * np.asarray(w_scales, np.float32).astype(np.float64)
    return prod.astype(np.float32)


def quantize_bias(b, input_scale, w_scales, rule="legacy"):
    """→ (int32 bias, float32 scale).
    rule="legacy": inv = float32(1.0 / double(scale)), q = round(float32(b × inv)), ±(2^31−1)로 자름
    rule="divide": q = round(double(b) / double(scale))"""
    s = bias_scales(input_scale, w_scales)
    if rule == "divide":
        q = quantize(b, s, 0, -(1 << 31), (1 << 31) - 1, axis=0 if s.size > 1 else None)
        return q.astype(np.int32), s
    inv = (1.0 / s.astype(np.float64)).astype(np.float32)
    prod = (np.asarray(b, np.float32) * inv).astype(np.float32)
    q = np.clip(round_half_away(prod), -((1 << 31) - 1), (1 << 31) - 1)
    return q.astype(np.int32), s


def quantize_activation(x, scale, zero_point):
    """추론 시 입력 양자화. TFLite QUANTIZE 연산(AffineQuantize)과 같이 float32 나눗셈 →
    반올림(동률은 0에서 먼 쪽) → zero-point 더하기 → int8 범위로 자르기."""
    t = np.asarray(x, np.float32) / np.float32(scale)
    q = round_half_away(t) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def quantize_input_reference_eval(x, scale, zero_point):
    """MLPerf Tiny 레퍼런스 평가 코드(eval_quantized_model.py)의 입력 변환을 그대로 옮긴 것:
    `np.array(dat/input_scale + input_zero_point, dtype=np.int8)` — 반올림 없이 0 쪽으로 버리고,
    범위도 자르지 않는다(int8 범위 밖 값은 C 변환 규칙대로 감긴다)."""
    v = np.asarray(x, np.float32) / np.float32(scale) + zero_point
    return np.array(v, dtype=np.int8)


def dequantize(q, scale, zero_point, axis=None):
    q = np.asarray(q, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    zero_point = np.asarray(zero_point, dtype=np.float64)
    if axis is not None and scale.size > 1:
        shape = [1] * q.ndim
        shape[axis] = -1
        scale, zero_point = scale.reshape(shape), np.broadcast_to(zero_point, scale.shape).reshape(shape)
    return (q - zero_point) * scale
