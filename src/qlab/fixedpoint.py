"""TFLite 정수 커널이 쓰는 고정소수점 연산을 numpy로 옮긴 것.

원문: LiteRT v2.2.0 `tflite/kernels/internal/common.cc`, `quantization_util.cc`,
gemmlowp `fixedpoint/fixedpoint.h`. C++의 int32 연산을 int64 배열에서 흉내 내므로,
각 함수는 입력이 int32 범위라는 전제를 확인하고 벗어나면 예외를 던진다.
"""
import math

import numpy as np

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def _i64(x):
    return np.asarray(x, dtype=np.int64)


def _check_int32(x, what):
    if np.any(x < INT32_MIN) or np.any(x > INT32_MAX):
        raise OverflowError(f"{what}: int32 범위를 벗어남")


def trunc_div(a, b):
    """C++ 정수 나눗셈(0 쪽으로 버림). numpy의 // 는 음수에서 내림이라 다르다."""
    a = _i64(a)
    q = np.abs(a) // b
    return np.where(a < 0, -q, q)


def round_half_away(x):
    """std::round — 반올림에서 .5는 0에서 먼 쪽으로. np.round(짝수 쪽)와 다르다."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ---------------------------------------------------------------- 배율 → 정수 곱셈자

def quantize_multiplier(real):
    """QuantizeMultiplier: 실수 배율을 (Q0.31 정수 곱셈자, 2의 지수)로 나눈다.

    real ≈ multiplier / 2^31 × 2^shift,  multiplier ∈ [2^30, 2^31)
    """
    real = float(real)
    if real == 0.0:
        return 0, 0
    q, shift = math.frexp(real)
    q_fixed = int(round_half_away(q * (1 << 31)))
    assert q_fixed <= (1 << 31)
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return q_fixed, shift


# ---------------------------------------------------------------- gemmlowp 기본 연산

def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul(int32, int32) = round(a·b / 2^31), 동률은 +∞ 쪽.
    ARM NEON VQRDMULH와 같은 연산이다(유일한 포화: a = b = INT32_MIN)."""
    a, b = _i64(a), _i64(b)
    _check_int32(a, "srdhm a")
    _check_int32(b, "srdhm b")
    ab = a * b
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    out = trunc_div(ab + nudge, 1 << 31)
    overflow = (a == b) & (a == INT32_MIN)
    return np.where(overflow, INT32_MAX, out)


def rounding_divide_by_pot(x, exponent):
    """RoundingDivideByPOT: 2^exponent 로 나누되 가장 가까운 정수로(동률은 0에서 먼 쪽)."""
    x, e = _i64(x), _i64(exponent)
    if np.any(e < 0) or np.any(e > 31):
        raise ValueError("exponent는 0..31")
    mask = (np.int64(1) << e) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (x < 0)
    return (x >> e) + (remainder > threshold)


def saturating_rounding_mul_by_pot(x, exponent):
    """SaturatingRoundingMultiplyByPOT<exponent> (int32)."""
    x = _i64(x)
    if exponent == 0:
        return x
    if exponent < 0:
        return rounding_divide_by_pot(x, -exponent)
    threshold = (1 << (31 - exponent)) - 1
    out = x << exponent
    out = np.where(x > threshold, INT32_MAX, out)
    return np.where(x < -threshold, INT32_MIN, out)


# ---------------------------------------------------------------- 재양자화

def mbqm_double(x, multiplier, shift):
    """MultiplyByQuantizedMultiplier, 기본 빌드. 두 번 반올림한다:
    SRDHM(동률은 +∞ 쪽) → RoundingDivideByPOT(동률은 0에서 먼 쪽)."""
    x, m, s = _i64(x), _i64(multiplier), _i64(shift)
    left = np.maximum(s, 0)
    right = np.maximum(-s, 0)
    shifted = x * (np.int64(1) << left)
    _check_int32(shifted, "x << left_shift")
    return rounding_divide_by_pot(srdhm(shifted, m), right)


def mbqm_single(x, multiplier, shift):
    """MultiplyByQuantizedMultiplier, TFLITE_SINGLE_ROUNDING 빌드. 정확한 곱을 한 번만
    반올림한다(동률은 +∞ 쪽). pip 휠이 어느 쪽인지는 실측으로 가린다(scripts/verify_bitexact.py)."""
    x, m, s = _i64(x), _i64(multiplier), _i64(shift)
    total = 31 - s
    rnd = np.int64(1) << (total - 1)
    out = (x * m + rnd) >> total
    _check_int32(out, "single-rounding 결과")
    return out


def mbqm_ruy_x86(x, multiplier, shift):
    """ruy의 x86 SIMD 커널(AVX2·AVX-512)이 하는 재양자화. 원문: ruy@3286a34 `kernel_avx512.cc`
    (TF v2.21.0이 고정한 커밋). 두 번 반올림하고 두 단계 모두 동률을 +∞ 쪽으로 올린다.
    기본 빌드와는 둘째 단계(기본 빌드는 동률을 0에서 먼 쪽으로)의 음수 동률에서만 다르다.

        v = ((x << left) · M + 2^30) >> 31
        v = (v + 2^(right−1)) >> right        (right > 0일 때)
    """
    x, m, s = _i64(x), _i64(multiplier), _i64(shift)
    left = np.maximum(s, 0)
    right = np.maximum(-s, 0)
    xl = x << left
    _check_int32(xl, "x << left_shift")
    v = (xl * m + (1 << 30)) >> 31
    nudge = np.where(right > 0, np.int64(1) << np.maximum(right - 1, 0), 0)
    overflow = v > INT32_MAX - nudge
    return np.where(overflow, np.int64(1) << (31 - right), (v + nudge) >> right)


def mbqm_neon(x, multiplier, shift):
    """LiteRT 최적화 커널의 per-channel `optimized_ops::Quantize`(기본 빌드) 중 NEON 구간.
    x86에서는 NEON_2_SSE 헤더가 같은 연산을 SSE로 흉내 낸다.

        v = vqrdmulhq(x << left, M)        = SRDHM (동률 +∞ 쪽)
        v = vrshlq(v, −right)              = (v + 2^(right−1)) >> right (동률 +∞ 쪽)
    """
    x, m, s = _i64(x), _i64(multiplier), _i64(shift)
    left = np.maximum(s, 0)
    right = np.maximum(-s, 0)
    v = srdhm(x << left, m)
    nudge = np.where(right > 0, np.int64(1) << np.maximum(right - 1, 0), 0)
    return (v + nudge) >> right


def requant_float(x, effective_scale):
    """LiteRT 참조 FC 커널(`reference/integer_ops/fully_connected.h`)의 재양자화:
    고정소수점 대신 round(double(acc) × double 배율). 반올림은 std::round(동률은 0에서 먼 쪽)."""
    return round_half_away(np.asarray(x, np.float64) * np.asarray(effective_scale, np.float64)
                           ).astype(np.int64)


REQUANT = {"double": mbqm_double, "single": mbqm_single, "ruy": mbqm_ruy_x86, "neon": mbqm_neon}


# ---------------------------------------------------------------- softmax용 (gemmlowp)

def _fp_mul(a, b):
    """FixedPoint 곱: 정수부 비트 수는 더해지고, 원시값은 SRDHM."""
    return srdhm(a, b)


def exp_on_interval_between_negative_one_quarter_and_0_excl(a):
    """Q0.31 입력 a ∈ [-1/4, 0) 에 대해 exp(a)를 Q0.31로. -1/8 근처 테일러 전개."""
    a = _i64(a)
    constant_term = 1895147668          # exp(-1/8)
    constant_1_over_3 = 715827883       # 1/3
    x = a + (1 << 28)                   # a + 1/8
    x2 = _fp_mul(x, x)
    x3 = _fp_mul(x2, x)
    x4 = _fp_mul(x2, x2)
    x4_over_4 = saturating_rounding_mul_by_pot(x4, -2)
    poly = saturating_rounding_mul_by_pot(
        _fp_mul(x4_over_4 + x3, constant_1_over_3) + x2, -1)
    return constant_term + _fp_mul(constant_term, x + poly)


def exp_on_negative_values(a, integer_bits):
    """exp(a), a ≤ 0 을 Q(integer_bits).(31-integer_bits) 입력에서 Q0.31 출력으로."""
    a = _i64(a)
    frac_bits = 31 - integer_bits
    one_quarter = 1 << (frac_bits - 2)
    mask = one_quarter - 1
    a_mod_q_minus_q = (a & mask) - one_quarter
    result = exp_on_interval_between_negative_one_quarter_and_0_excl(
        saturating_rounding_mul_by_pot(a_mod_q_minus_q, integer_bits))  # Rescale<0>
    remainder = a_mod_q_minus_q - a
    for exponent, mult in ((-2, 1672461947), (-1, 1302514674), (0, 790015084),
                           (1, 290630308), (2, 39332535), (3, 720401), (4, 242)):
        if integer_bits > exponent:
            bit = np.int64(1) << (frac_bits + exponent)
            result = np.where((remainder & bit) != 0, _fp_mul(result, mult), result)
    if integer_bits > 5:
        clamp = -(1 << (36 - integer_bits))
        result = np.where(a < clamp, 0, result)
    return np.where(a == 0, INT32_MAX, result)   # One() of Q0.31


def one_over_one_plus_x_for_x_in_0_1(a):
    """1/(1+a), a ∈ [0,1) 를 Q0.31로. 뉴턴-랩슨 3회."""
    a = _i64(a)
    half_denominator = trunc_div(a + INT32_MAX + np.where(a + INT32_MAX >= 0, 1, -1), 2)
    c48_17 = 1515870810      # Q2.29
    c_neg32_17 = -1010580540  # Q2.29
    one_q2 = 1 << 29
    x = c48_17 + _fp_mul(half_denominator, c_neg32_17)
    for _ in range(3):
        hd_x = _fp_mul(half_denominator, x)
        one_minus = one_q2 - hd_x
        x = x + saturating_rounding_mul_by_pot(_fp_mul(x, one_minus), 2)  # Rescale<2> from Q4
    return saturating_rounding_mul_by_pot(x, 1)   # Rescale<0>(ExactMulByPot<-1>(x)): Q1 → Q0


def count_leading_zeros_u32(x):
    x = _i64(x) & 0xFFFFFFFF
    _, bit_length = np.frexp(x.astype(np.float64))   # uint32는 float64로 정확히 표현된다
    return np.where(x == 0, 32, 32 - bit_length).astype(np.int64)


def get_reciprocal(x, x_integer_digits):
    """GetReciprocal: x(Q x_integer_digits)의 역수를 Q0.31 배율과 비트 수로."""
    x = _i64(x)
    headroom_plus_one = count_leading_zeros_u32(x)
    num_bits_over_unit = x_integer_digits - headroom_plus_one
    shifted = ((x << headroom_plus_one) & 0xFFFFFFFF) - (1 << 31)
    return one_over_one_plus_x_for_x_in_0_1(shifted), num_bits_over_unit


def preprocess_softmax_scaling(beta, input_scale, input_integer_bits):
    """PreprocessSoftmaxScaling(기본 빌드): beta·scale 을 Q(input_integer_bits) 곱셈자로."""
    max_real_multiplier = (1 << 31) - 1.0
    input_beta_real_multiplier = min(
        beta * input_scale * (1 << (31 - input_integer_bits)), max_real_multiplier)
    multiplier, shift = quantize_multiplier(input_beta_real_multiplier)   # GreaterThanOne
    assert shift >= 0
    return multiplier, shift


def calculate_input_radius(input_integer_bits, input_left_shift, total_signed_bits=31):
    max_input_rescaled = (1.0 * ((1 << input_integer_bits) - 1)
                          * (1 << (total_signed_bits - input_integer_bits))
                          / (1 << input_left_shift))
    return int(math.floor(max_input_rescaled))
