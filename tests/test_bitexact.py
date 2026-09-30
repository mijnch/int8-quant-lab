"""T2·T3 — numpy 커널 대 LiteRT 세 실행 경로(참조·최적화·XNNPACK), 비트 단위 비교."""
import numpy as np
import pytest

from conftest import MODELS
from qlab import runtime as R
from qlab import synth
from qlab.model import Model

RUNTIMES = ("reference", "optimized", "xnnpack")
INT8_MODELS = ("kws_ref_model.tflite", "kws_tf221_int8.tflite")


@pytest.fixture(scope="module")
def inputs():
    return np.random.default_rng(0).integers(-128, 128, size=(64, 49, 10, 1), dtype=np.int8)


@pytest.mark.parametrize("name", INT8_MODELS)
@pytest.mark.parametrize("mode", RUNTIMES)
def test_every_layer_matches(name, mode, inputs):
    path = MODELS / name
    m = Model(path)
    it = R.make(path, mode, batch=len(inputs), preserve=True, split_delegate=(mode == "xnnpack"))
    R.invoke(it, inputs)
    for op in m.ops:
        mine = m.run_op(op, it.get_tensor(op.inputs[0]), policy=mode)
        assert np.array_equal(mine, it.get_tensor(op.outputs[0])), f"{op.index}:{op.kind}"


@pytest.mark.parametrize("name", INT8_MODELS)
@pytest.mark.parametrize("mode", RUNTIMES)
def test_whole_network_matches(name, mode, inputs):
    path = MODELS / name
    assert np.array_equal(Model(path).run(inputs, policy=mode), R.run(path, mode, inputs))


@pytest.mark.parametrize("kind", ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
                                  "AVERAGE_POOL_2D", "SOFTMAX"])
def test_random_single_op_models(kind):
    rng = np.random.default_rng(123)
    for _ in range(8):
        blob = synth.random_model(rng, kind)
        m = Model(blob)
        shape = m.tensors[m.inputs[0]].shape
        x = rng.integers(-128, 128, size=(32,) + tuple(shape[1:]), dtype=np.int8)
        for mode in RUNTIMES:
            lib = R.invoke(R.make(blob, mode, batch=len(x)), x)
            assert np.array_equal(m.run_op(m.ops[0], x, policy=mode), lib), mode
