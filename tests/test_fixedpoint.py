"""고정소수점 기본 연산 — 알려진 값, 동률 처리, 정확한 유리수 계산과의 비교."""
import math

import numpy as np

from qlab import fixedpoint as fx


def test_quantize_multiplier_known_values():
    assert fx.quantize_multiplier(0.0) == (0, 0)
    assert fx.quantize_multiplier(0.5) == (1 << 30, 0)
    assert fx.quantize_multiplier(1.0) == (1 << 30, 1)
    assert fx.quantize_multiplier(0.75) == (3 << 29, 0)
    assert fx.quantize_multiplier(1e-12) == (0, 0)             # 2^-31보다 작으면 0으로
    assert fx.quantize_multiplier(1 - 2 ** -40) == (1 << 30, 1)  # 2^31로 반올림되면 절반+지수 1


def test_srdhm_saturation_and_ties():
    assert fx.srdhm(fx.INT32_MIN, fx.INT32_MIN) == fx.INT32_MAX
    assert fx.srdhm(1 << 30, 1 << 30) == 1 << 29
    # round(a·b/2^31)의 동률은 +∞ 쪽: -0.5 → 0, -1.5 → -1, 0.5 → 1
    assert fx.srdhm(-1, 1 << 30) == 0
    assert fx.srdhm(-3, 1 << 30) == -1
    assert fx.srdhm(1, 1 << 30) == 1


def test_rounding_divide_by_pot_ties_away_from_zero():
    assert fx.rounding_divide_by_pot(3, 1) == 2
    assert fx.rounding_divide_by_pot(-3, 1) == -2
    assert fx.rounding_divide_by_pot(-5, 2) == -1               # -1.25
    assert fx.rounding_divide_by_pot(-6, 2) == -2               # -1.5


def test_requant_variants_differ_only_at_negative_ties():
    m, s = 1 << 30, -1                                         # 배율 0.25
    assert fx.mbqm_double(-6, m, s) == -2                      # -1.5 → 0에서 먼 쪽
    assert fx.mbqm_single(-6, m, s) == -1                      # -1.5 → +∞ 쪽
    assert fx.mbqm_ruy_x86(-6, m, s) == -1
    for f in (fx.mbqm_double, fx.mbqm_single, fx.mbqm_ruy_x86):
        assert f(6, m, s) == 2


def test_single_rounding_is_exact_round_half_up():
    rng = np.random.default_rng(0)
    x = rng.integers(-(1 << 24), 1 << 24, 2000)
    m = rng.integers(1 << 30, (1 << 31) - 1, 2000)
    s = rng.integers(-12, 1, 2000)
    got = fx.mbqm_single(x, m, s)
    for xi, mi, si, g in zip(x.tolist(), m.tolist(), s.tolist(), got.tolist()):
        total = 31 - si
        assert g == (xi * mi + (1 << (total - 1))) >> total       # 파이썬 정수로 정확히 계산


def test_double_rounding_error_bounded():
    rng = np.random.default_rng(1)
    x = rng.integers(-(1 << 24), 1 << 24, 5000)
    m = rng.integers(1 << 30, (1 << 31) - 1, 5000)
    s = rng.integers(-12, 1, 5000)
    exact = x * (m / 2.0 ** 31) * 2.0 ** s
    assert np.abs(fx.mbqm_double(x, m, s) - exact).max() <= 1.0


def test_gemmlowp_exp_and_reciprocal_accuracy():
    a = -np.arange(0, 1 << 31, 1 << 20, dtype=np.int64)          # Q5.26, [-32, 0]
    got = fx.exp_on_negative_values(a, 5) / 2 ** 31
    assert np.abs(got - np.exp(a / 2 ** 26)).max() < 1e-6
    x = np.arange(0, 1 << 31, 1 << 16, dtype=np.int64)          # Q0.31, [0, 1)
    got = fx.one_over_one_plus_x_for_x_in_0_1(x) / 2 ** 31
    assert np.abs(got - 1 / (1 + x / 2 ** 31)).max() < 1e-8


def test_count_leading_zeros():
    v = np.array([0, 1, 2, 3, (1 << 31) - 1, 1 << 31, (1 << 32) - 1])
    assert fx.count_leading_zeros_u32(v).tolist() == [32, 31, 30, 30, 1, 0, 0]


def test_quantize_multiplier_reconstructs_scale():
    for real in (0.0123, 0.5, 0.999, 3.7e-5, 1.5, 200.0):
        m, s = fx.quantize_multiplier(real)
        assert math.isclose(m / 2 ** 31 * 2 ** s, real, rel_tol=2 ** -30)
