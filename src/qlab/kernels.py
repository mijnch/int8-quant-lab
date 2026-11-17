"""T2·T3 — int8 연산자를 TFLite 참조 커널과 같은 정수 산술로 계산한다(NHWC).

공통 흐름(원문: LiteRT v2.2.0 `reference/integer_ops/{conv,depthwise_conv,fully_connected}.h`):

    acc  = Σ (x − x_zp) · w   (+ bias)          int32 누산
    acc  = 재양자화(acc, 채널별 배율)             실수 배율 s_in·s_w/s_out 곱하기
    out  = clamp(acc + y_zp, act_min, act_max)   int8

재양자화는 런타임마다 구현이 다르다(_requant_clamp 참고). 어느 런타임이 어느 방식을 쓰는지는
model.POLICIES에 원문 근거와 함께 정리했고, scripts/verify_bitexact.py가 실측으로 확인한다.

누산은 float64 행렬곱으로 한다. |x − zp| ≤ 255, |w| ≤ 127, 곱한 항이 K개이면
|acc| ≤ 255·127·K 이고, 이 값이 2^53보다 작으면 float64 합은 정확한 정수다(아래에서 확인).
"""
from dataclasses import dataclass

import numpy as np

from . import fixedpoint as fx

_EXACT = 1 << 53


def _exact_matmul(a, b):
    """정수 행렬 a @ b를 float64 BLAS로 계산하되, 결과가 정확함을 보장한다."""
    bound = np.abs(a).max(initial=0) * np.abs(b).max(initial=0) * a.shape[-1]
    if bound >= _EXACT:
        raise OverflowError("누산 범위가 float64 정수 표현 한계를 넘는다")
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.int64)


@dataclass
class Requant:
    """출력 채널별 재양자화 파라미터. 같은 배율을 구현마다 다른 형태로 쓴다."""
    mult: np.ndarray     # Q0.31 정수 곱셈자 (TFLite·ruy)
    shift: np.ndarray    # 2의 지수
    eff: np.ndarray      # double 배율 (LiteRT 참조 FC)
    eff32: np.ndarray    # float32 배율 (XNNPACK)

    def repeat(self, n):
        return Requant(*(np.repeat(a, n) for a in (self.mult, self.shift, self.eff, self.eff32)))


def conv_multipliers(input_scale, filter_scales, output_scale):
    """채널별 재양자화 파라미터.
    - TFLite(PopulateConvolutionQuantizationParams): eff = double(s_in)·double(s_w)/double(s_out)
      → QuantizeMultiplier로 (M, shift)
    - XNNPACK(convolution-nhwc.c): float32로 s_in·s_w/s_out"""
    s_in, s_out = np.float32(input_scale), np.float32(output_scale)
    mult, shift, effs, effs32 = [], [], [], []
    for s_w in np.atleast_1d(np.asarray(filter_scales, np.float32)):
        eff = float(s_in) * float(s_w) / float(s_out)
        m, sh = fx.quantize_multiplier(eff)
        mult.append(m)
        shift.append(sh)
        effs.append(eff)
        effs32.append(np.float32(np.float32(s_in * s_w) / s_out))
    return Requant(np.array(mult, np.int64), np.array(shift, np.int64),
                   np.array(effs, np.float64), np.array(effs32, np.float32))


def activation_range(fused, output_scale, output_zp):
    """CalculateActivationRangeQuantized (int8). fused ∈ {NONE, RELU, RELU6, RELU_N1_TO_1}."""
    def q(f):
        return output_zp + int(fx.round_half_away(np.float32(f) / np.float32(output_scale)))
    lo, hi = -128, 127
    if fused == "RELU":
        lo = max(lo, q(0.0))
    elif fused == "RELU6":
        lo, hi = max(lo, q(0.0)), min(hi, q(6.0))
    elif fused == "RELU_N1_TO_1":
        lo, hi = max(lo, q(-1.0)), min(hi, q(1.0))
    elif fused != "NONE":
        raise NotImplementedError(fused)
    return lo, hi


ROUNDINGS = ("double", "single", "ruy", "float", "fp32", "neon8")


def _requant_clamp(acc, rq, out_zp, act_min, act_max, rounding):
    """재양자화 → zero-point 더하기 → 활성 범위로 자르기.

    double : 기본 빌드 TFLite 고정소수점(SRDHM[동률 +∞ 쪽] → RoundingDivideByPOT[동률 0에서 먼 쪽])
    single : TFLITE_SINGLE_ROUNDING 빌드(정확한 곱을 한 번 반올림, 동률은 +∞ 쪽)
    ruy    : ruy x86 SIMD(두 번 반올림, 두 단계 모두 동률은 +∞ 쪽)
    float  : round(double(acc) × double 배율) — LiteRT 참조 FC
    fp32   : rint(float32(acc) × float32 배율), 동률은 짝수 쪽 — XNNPACK qs8 "fp32" 커널
    neon8  : LiteRT optimized_ops::Quantize — 채널을 8개씩 묶은 앞 구간(8·⌊C/8⌋개)은 NEON 경로
             (SRDHM → 동률 +∞ 쪽 반올림 시프트), 나머지 채널은 double과 같은 스칼라 경로
    """
    if rounding == "float":
        scaled = fx.requant_float(acc, rq.eff)
        fx._check_int32(scaled, "float 재양자화 결과")
    elif rounding == "neon8":
        k = (acc.shape[-1] // 8) * 8
        scaled = np.concatenate([fx.mbqm_neon(acc[..., :k], rq.mult[:k], rq.shift[:k]),
                                 fx.mbqm_double(acc[..., k:], rq.mult[k:], rq.shift[k:])], axis=-1)
    elif rounding == "fp32":
        v = acc.astype(np.float32) * rq.eff32                   # float32 곱
        v = np.minimum(v, np.float32(act_max - out_zp))
        scaled = np.rint(v).astype(np.int64)                    # 가장 가까운 짝수로
    else:
        scaled = fx.REQUANT[rounding](acc, rq.mult, rq.shift)
    return np.clip(scaled + out_zp, act_min, act_max).astype(np.int8)


def same_padding(in_size, filter_size, stride, dilation=1):
    """ComputePaddingWithOffset (padding.h): (앞쪽 패딩, 뒤쪽 패딩, 출력 크기)."""
    out = (in_size + stride - 1) // stride
    eff = (filter_size - 1) * dilation + 1
    total = max((out - 1) * stride + eff - in_size, 0)
    return total // 2, total - total // 2, out


def _patches(x, kh, kw, sh, sw, dh, dw, padding, pad_value):
    """x[N,H,W,C] → 패치[N,OH,OW,kh,kw,C]. 패딩 칸은 pad_value(= 입력 zero-point)로 채워
    (x − zp) = 0 이 되게 한다 — 참조 커널이 패딩 칸을 건너뛰는 것과 같은 결과."""
    n, h, w, c = x.shape
    if padding == "SAME":
        pt, pb, oh = same_padding(h, kh, sh, dh)
        pl, pr, ow = same_padding(w, kw, sw, dw)
    elif padding == "VALID":
        pt = pb = pl = pr = 0
        oh = (h - ((kh - 1) * dh + 1)) // sh + 1
        ow = (w - ((kw - 1) * dw + 1)) // sw + 1
    else:
        raise NotImplementedError(padding)
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)), constant_values=pad_value)
    idx_h = (np.arange(oh) * sh)[:, None] + np.arange(kh) * dh          # [OH, kh]
    idx_w = (np.arange(ow) * sw)[:, None] + np.arange(kw) * dw          # [OW, kw]
    return xp[:, idx_h[:, None, :, None], idx_w[None, :, None, :], :]   # [N,OH,OW,kh,kw,C]


def conv2d(x, w, b, *, in_zp, out_zp, rq, stride, padding, act_range,
           dilation=(1, 1), rounding="double"):
    """CONV_2D. x int8[N,H,W,Ci], w int8[Co,kh,kw,Ci], b int32[Co] 또는 None."""
    co, kh, kw, ci = w.shape
    p = _patches(x.astype(np.int64), kh, kw, *stride, *dilation, padding, in_zp) - in_zp
    n, oh, ow = p.shape[:3]
    acc = _exact_matmul(p.reshape(n * oh * ow, kh * kw * ci), w.reshape(co, -1).T.astype(np.int64))
    if b is not None:
        acc = acc + b.astype(np.int64)
    out = _requant_clamp(acc, rq, out_zp, *act_range, rounding)
    return out.reshape(n, oh, ow, co)


def depthwise_conv2d(x, w, b, *, in_zp, out_zp, rq, stride, padding, act_range,
                     depth_multiplier=1, dilation=(1, 1), rounding="double"):
    """DEPTHWISE_CONV_2D. w int8[1,kh,kw,Ci·dm]; 출력 채널 oc = ic·dm + m.
    필터 칸(kh·kw)마다 입력을 밀어 곱해 더한다 — 패치 배열을 만들지 않아 메모리가 적게 든다."""
    _, kh, kw, co = w.shape
    n, h, wd, ci = x.shape
    assert co == ci * depth_multiplier
    (sh, sw), (dh, dw) = stride, dilation
    if padding == "SAME":
        pt, pb, oh = same_padding(h, kh, sh, dh)
        pl, pr, ow = same_padding(wd, kw, sw, dw)
    else:
        pt = pb = pl = pr = 0
        oh = (h - ((kh - 1) * dh + 1)) // sh + 1
        ow = (wd - ((kw - 1) * dw + 1)) // sw + 1
    xp = np.pad(x.astype(np.int64) - in_zp, ((0, 0), (pt, pb), (pl, pr), (0, 0)))  # 패딩 칸 = 0
    xp = np.repeat(xp, depth_multiplier, axis=-1)
    wi = w[0].astype(np.int64)
    acc = np.zeros((n, oh, ow, co), np.int64)
    for i in range(kh):
        for j in range(kw):
            acc += xp[:, i * dh: i * dh + sh * (oh - 1) + 1: sh,
                      j * dw: j * dw + sw * (ow - 1) + 1: sw, :] * wi[i, j]
    if b is not None:
        acc = acc + b.astype(np.int64)
    return _requant_clamp(acc, rq, out_zp, *act_range, rounding)


def fully_connected(x, w, b, *, in_zp, out_zp, rq, act_range, rounding="double"):
    """FULLY_CONNECTED. x int8[N,K], w int8[Co,K]."""
    acc = _exact_matmul(x.astype(np.int64) - in_zp, w.T.astype(np.int64))
    if b is not None:
        acc = acc + b.astype(np.int64)
    return _requant_clamp(acc, rq, out_zp, *act_range, rounding)


def average_pool(x, *, filter_hw, stride, padding, act_range):
    """AVERAGE_POOL_2D (int8, 참조 커널). 입력과 출력의 양자화 파라미터가 같아야 한다.
    합을 유효 칸 수로 나누되 가장 가까운 정수로(동률은 0에서 먼 쪽)."""
    kh, kw = filter_hw
    n, h, w, c = x.shape
    if padding == "VALID":
        pt = pl = 0
        oh = (h - kh) // stride[0] + 1
        ow = (w - kw) // stride[1] + 1
    else:
        pt, _, oh = same_padding(h, kh, stride[0])
        pl, _, ow = same_padding(w, kw, stride[1])
    out = np.empty((n, oh, ow, c), np.int8)
    xi = x.astype(np.int64)
    for oy in range(oh):
        for ox in range(ow):
            y0, x0 = oy * stride[0] - pt, ox * stride[1] - pl
            ys, xs = max(0, y0), max(0, x0)
            ye, xe = min(h, y0 + kh), min(w, x0 + kw)
            win = xi[:, ys:ye, xs:xe, :]
            count = (ye - ys) * (xe - xs)
            acc = win.sum(axis=(1, 2))
            acc = np.where(acc > 0, fx.trunc_div(acc + count // 2, count),
                           fx.trunc_div(acc - count // 2, count))
            out[:, oy, ox, :] = np.clip(acc, *act_range)
    return out


# ------------------------------------------------------------------ softmax (int8 → int8)

def _load_expf():
    """C 라이브러리의 expf를 그대로 쓴다(런타임이 부르는 함수와 같게). 못 찾으면
    double exp를 float32로 반올림한 값으로 대신한다 — 드물게 1 ulp 다를 수 있다."""
    import ctypes
    import ctypes.util
    for name in (ctypes.util.find_library("m"), "libm.so.6", "msvcrt", "ucrtbase"):
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            f = lib.expf
        except (OSError, AttributeError):
            continue
        f.restype, f.argtypes = ctypes.c_float, [ctypes.c_float]
        return (lambda v: np.float32(f(float(v)))), name
    return (lambda v: np.float32(np.exp(np.float64(v)))), None


_expf, EXPF_SOURCE = _load_expf()


def softmax_reference(x, *, input_scale, beta=1.0):
    """참조 커널(BUILTIN_REF)의 int8 softmax: gemmlowp 고정소수점 exp와 역수.
    원문: LiteRT `reference/softmax.h`, `activations.cc`(SoftmaxPrepare, kReference)."""
    k_scaled_diff_bits, k_accum_bits = 5, 12
    mult, left_shift = fx.preprocess_softmax_scaling(float(np.float32(beta)),
                                                     float(np.float32(input_scale)),
                                                     k_scaled_diff_bits)
    diff_min = -fx.calculate_input_radius(k_scaled_diff_bits, left_shift)
    x = x.astype(np.int64)
    diff = x - x.max(axis=-1, keepdims=True)
    use = diff >= diff_min
    # MultiplyByQuantizedMultiplierGreaterThanOne(기본 빌드) = SRDHM(x << left_shift, M)
    rescaled = fx.srdhm(np.where(use, diff, 0) << left_shift, mult)
    exp_q0 = fx.exp_on_negative_values(rescaled, k_scaled_diff_bits)          # Q0.31
    exp_acc = fx.saturating_rounding_mul_by_pot(exp_q0, -k_accum_bits)        # Rescale<12>
    sum_exp = np.where(use, exp_acc, 0).sum(axis=-1, keepdims=True)
    fx._check_int32(sum_exp, "softmax 합")
    shifted_scale, bits_over_unit = fx.get_reciprocal(sum_exp, k_accum_bits)
    exponent = bits_over_unit + 31 - 8
    assert np.all((exponent >= 0) & (exponent <= 31))
    unsat = fx.rounding_divide_by_pot(fx.srdhm(shifted_scale, exp_q0), exponent)
    out = np.clip(unsat - 128, -128, 127)
    return np.where(use, out, -128).astype(np.int8)


def softmax_optimized(x, *, input_scale, beta=1.0, output_scale=1 / 256, output_zp=-128):
    """최적화 커널(BUILTIN, x86)의 int8 softmax: float32 exp 표(256칸)를 쓴다.
    원문: LiteRT `optimized/optimized_ops.h` PopulateSoftmaxLookupTable, Softmax<In,Out>."""
    scale = np.float32(-np.float32(input_scale) * np.float32(beta))
    table = np.empty(256, np.float32)
    for val in range(256):                         # table[255 - val] = expf(scale · val)
        table[255 - val] = _expf(np.float32(scale * np.float32(val)))
    x = x.astype(np.int64)
    rows = x.reshape(-1, x.shape[-1])
    out = np.empty(rows.shape, np.int8)
    for i, row in enumerate(rows):
        offset = 255 - int(row.max())
        vals = table[offset + row]
        s = np.float32(0)
        for v in vals:                              # C++ 루프와 같은 순서로 float32 누적
            s = np.float32(s + v)
        inv = np.float32(np.float32(1.0) / np.float32(s * np.float32(output_scale)))
        prob = (vals * inv).astype(np.float32)
        q = fx.round_half_away(prob).astype(np.int64) + output_zp
        out[i] = np.clip(q, -128, 127)
    return out.reshape(x.shape)
