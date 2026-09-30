"""가중치만 양자화→역양자화한 float 모델을 만든다(fake quantization 시뮬레이션).

변환기는 가중치를 항상 대칭(zero-point 0)으로 양자화하므로, 비대칭 가중치나 4비트 같은 설계
변수는 변환기로 비교할 수 없다. 여기서는 float `.tflite`의 가중치 버퍼만 바꿔 끼워,
"가중치 양자화 오차가 정확도에 주는 영향"만 따로 잰다. 활성값은 float 그대로다 —
실제 int8 추론 결과가 아니라는 점을 결과 표에 구분해 적는다.
"""
import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S

from .fixedpoint import round_half_away

_WEIGHT_AXIS = {"CONV_2D": 0, "DEPTHWISE_CONV_2D": 3, "FULLY_CONNECTED": 0}
_OP_NAMES = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}


def fake_quant(w, bits, symmetric, axis):
    """w(float32)를 bits비트로 양자화했다가 되돌린 값. axis=None이면 텐서 전체에 scale 하나.

    대칭: 좁은 범위 [-(2^(b-1)-1), 2^(b-1)-1], scale = max|w| / (2^(b-1)-1), zero-point 0
    비대칭: [-2^(b-1), 2^(b-1)-1], 범위는 0을 포함하게 넓힌 [min, max],
            scale = (max-min)/(2^b-1), zero-point = round(qmin - min/scale) (범위로 자름)
    """
    w = np.asarray(w, np.float64)
    red = None if axis is None else tuple(i for i in range(w.ndim) if i != axis)
    keep = dict(axis=red, keepdims=True)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scale = np.abs(w).max(**keep) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        q = np.clip(round_half_away(w / scale), -qmax, qmax)
        return (q * scale).astype(np.float32)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    lo = np.minimum(w.min(**keep), 0.0)
    hi = np.maximum(w.max(**keep), 0.0)
    scale = (hi - lo) / (qmax - qmin)
    scale = np.where(scale == 0, 1.0, scale)
    zp = np.clip(round_half_away(qmin - lo / scale), qmin, qmax)
    q = np.clip(round_half_away(w / scale) + zp, qmin, qmax)
    return ((q - zp) * scale).astype(np.float32)


def sqnr_db(w, wq):
    w = np.asarray(w, np.float64)
    noise = np.sum((w - wq) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(np.sum(w ** 2) / noise))


def quantize_weights_in_model(float_tflite, bits, symmetric, per_channel, ops=None):
    """float `.tflite`의 CONV·DEPTHWISE·FC 가중치를 fake_quant로 바꾼 새 모델 bytes와,
    층별 가중치 SQNR(dB)을 돌려준다. bias는 float 그대로 둔다.
    ops: 양자화할 연산 번호의 집합(None이면 전부) — 층별 민감도 분석용."""
    m = S.ModelT.InitFromPackedBuf(float_tflite, 0)
    sg = m.subgraphs[0]
    sqnr = []
    for i, op in enumerate(sg.operators):
        oc = m.operatorCodes[op.opcodeIndex]
        kind = _OP_NAMES[max(oc.builtinCode, oc.deprecatedBuiltinCode)]
        if kind not in _WEIGHT_AXIS or (ops is not None and i not in ops):
            continue
        t = sg.tensors[op.inputs[1]]
        if t.type != S.TensorType.FLOAT32:
            raise ValueError("가중치가 float32인 모델이어야 한다(순수 float 변환본)")
        buf = m.buffers[t.buffer]
        w = np.frombuffer(bytes(bytearray(buf.data)), np.float32).reshape(tuple(t.shape))
        wq = fake_quant(w, bits, symmetric, _WEIGHT_AXIS[kind] if per_channel else None)
        sqnr.append(sqnr_db(w, wq))
        buf.data = np.frombuffer(wq.tobytes(), np.uint8)
    fb = flatbuffers.Builder(1024)
    fb.Finish(m.Pack(fb), file_identifier=b"TFL3")
    return bytes(fb.Output()), sqnr
