"""T1 — 저장소에 넣어 둔 TF 2.21 변환 결과(보정 모델 + int8 모델)를 명세 규칙으로 재현한다."""
import numpy as np

from conftest import MODELS
from qlab import spec, t1
from qlab.model import Model


def test_activation_qparams_rule():
    s, z = spec.activation_qparams(0.0, 25.5)
    assert s == np.float32(0.1) and z == -128
    s, z = spec.activation_qparams(-1.0, 1.0)                  # zp = round(-0.5) = -1
    assert s == np.float32(2.0) / np.float32(255) and z == -1
    s, z = spec.activation_qparams(0.5, 2.0)                   # 0을 포함하도록 min을 0으로
    assert z == -128 and s == np.float32(2.0) / np.float32(255)


def test_weight_scales_symmetric_per_channel():
    w = np.array([[1.0, -2.0], [0.5, 0.25]], np.float32)
    s = spec.weight_scales(w, axis=0)
    assert np.array_equal(s, np.array([4.0, 1.0], np.float32) / np.float32(254))
    q, _ = spec.quantize_weights(w, axis=0)
    assert q.tolist() == [[64, -127], [127, 64]]


def test_converter_output_is_reproduced_exactly():
    r = t1.verify(Model(MODELS / "kws_tf221_calibrated.tflite"), Model(MODELS / "kws_tf221_int8.tflite"))
    s = r["summary"]
    assert s["activation_qparams_equal"] == "14/14"
    assert s["weight_scales_equal"] == "10/10" and s["bias_scales_equal"] == "10/10"
    assert s["weight_int8_mismatch"] == 0 and s["weight_count"] == 22016
    assert s["bias_int32_mismatch"] == 0 and s["bias_count"] == 588
    # 나눗셈 규칙(일반 MLIR 경로)으로는 bias 하나가 1 차이 난다 — 역수 곱 규칙이 필요한 이유
    assert s["bias_int32_mismatch_divide_rule"] == 1
