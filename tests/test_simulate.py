"""simulate.map_weights — 가중치 버퍼만 바꿔 끼운 모델이 바뀐 가중치로 실제로 돈다."""
import numpy as np

from conftest import MODELS
from qlab import runtime as R
from qlab import simulate
from qlab.model import Model
from qlab.simulate import _WEIGHT_AXIS, fake_quant

FLOAT = MODELS / "kws_tf221_float32.tflite"


def _weights(blob):
    m = Model(blob)
    return {op.index: m.tensors[op.inputs[1]].data for op in m.ops if op.kind in _WEIGHT_AXIS}


def test_map_weights_replaces_only_requested_layers():
    base = FLOAT.read_bytes()
    before = _weights(base)
    seen = []

    def halve(i, kind, w):
        seen.append((i, kind))
        return w * 0.5

    after = _weights(simulate.map_weights(base, halve, ops={6}))
    assert seen == [(6, "CONV_2D")]
    for i in before:
        expect = before[i] * 0.5 if i == 6 else before[i]
        assert np.array_equal(after[i], expect)


def test_map_weights_identity_keeps_outputs():
    base = FLOAT.read_bytes()
    same = simulate.map_weights(base, lambda i, k, w: w)
    x = np.random.default_rng(0).standard_normal((4, 49, 10, 1)).astype(np.float32)
    assert np.array_equal(R.run(base, "optimized", x), R.run(same, "optimized", x))


def test_quantize_weights_in_model_matches_direct_fake_quant():
    base = FLOAT.read_bytes()
    blob, sqnr = simulate.quantize_weights_in_model(base, 4, True, False)
    got = _weights(blob)
    for i, w in _weights(base).items():
        assert np.array_equal(got[i], fake_quant(w, 4, True, None))
    assert len(sqnr) == len(got) and all(np.isfinite(sqnr))


def test_map_weights_rejects_shape_change():
    base = FLOAT.read_bytes()
    try:
        simulate.map_weights(base, lambda i, k, w: w.ravel(), ops={6})
    except ValueError as e:
        assert "모양" in str(e)
    else:
        raise AssertionError("모양이 바뀐 가중치를 받아들였다")
