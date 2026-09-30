"""검증용 단일 연산 int8 `.tflite` 모델을 직접 만든다(변환기·TensorFlow 없이).

실제 모델은 ReLU가 음수 결과를 잘라내 재양자화 방식의 차이가 가려진다. 활성 함수·scale·
zero-point·보폭·패딩을 무작위로 바꾼 연산을 세 런타임에서 돌려 대조하면 그 차이까지 드러난다.
"""
import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as S

_VERSIONS = {"CONV_2D": 3, "DEPTHWISE_CONV_2D": 3, "FULLY_CONNECTED": 4,
             "AVERAGE_POOL_2D": 2, "SOFTMAX": 2}
_ACTS = {"NONE": S.ActivationFunctionType.NONE, "RELU": S.ActivationFunctionType.RELU,
         "RELU6": S.ActivationFunctionType.RELU6}
_PADS = {"SAME": S.Padding.SAME, "VALID": S.Padding.VALID}


class _Builder:
    def __init__(self):
        self.tensors, self.buffers = [], [S.BufferT()]      # 0번 버퍼는 빈 자리표시

    def tensor(self, name, shape, ttype, scale, zp, axis=0, data=None):
        t = S.TensorT()
        t.name, t.shape, t.type = name.encode(), np.array(shape, np.int32), ttype
        q = S.QuantizationParametersT()
        q.scale = [float(s) for s in np.atleast_1d(scale)]
        q.zeroPoint = [int(z) for z in np.atleast_1d(zp)]
        q.quantizedDimension = axis
        t.quantization = q
        if data is None:
            t.buffer = 0
        else:
            b = S.BufferT()
            b.data = np.frombuffer(np.ascontiguousarray(data).tobytes(), np.uint8)
            self.buffers.append(b)
            t.buffer = len(self.buffers) - 1
        self.tensors.append(t)
        return len(self.tensors) - 1

    def finish(self, kind, inputs, output, options_type, options):
        oc = S.OperatorCodeT()
        code = getattr(S.BuiltinOperator, kind)
        oc.builtinCode, oc.deprecatedBuiltinCode, oc.version = code, min(code, 127), _VERSIONS[kind]
        op = S.OperatorT()
        op.opcodeIndex, op.inputs, op.outputs = 0, np.array(inputs, np.int32), np.array([output], np.int32)
        op.builtinOptionsType, op.builtinOptions = options_type, options
        sg = S.SubGraphT()
        sg.tensors, sg.inputs, sg.outputs, sg.operators = self.tensors, np.array([0], np.int32), \
            np.array([output], np.int32), [op]
        m = S.ModelT()
        m.version, m.operatorCodes, m.subgraphs, m.buffers = 3, [oc], [sg], self.buffers
        m.description = b"qlab synthetic single-op model"
        fb = flatbuffers.Builder(1024)
        fb.Finish(m.Pack(fb), file_identifier=b"TFL3")
        return bytes(fb.Output())


def _bias(rng, n, in_scale, w_scales):
    """변환기와 같은 규칙의 bias scale(float32(s_in·s_w))과 적당한 크기의 int32 값."""
    b_scale = (np.float64(np.float32(in_scale)) * np.asarray(w_scales, np.float32).astype(np.float64)
               ).astype(np.float32)
    return rng.integers(-20000, 20000, n).astype(np.int32), b_scale


def conv2d(rng, *, depthwise=False, act="NONE", padding="SAME", stride=(1, 1)):
    h, w_, ci = int(rng.integers(3, 12)), int(rng.integers(3, 12)), int(rng.integers(1, 21))
    kh, kw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    co = ci if depthwise else int(rng.integers(1, 17))
    in_s, in_zp = float(rng.uniform(0.005, 0.1)), int(rng.integers(-128, 128))
    out_s, out_zp = float(rng.uniform(0.02, 0.5)), int(rng.integers(-128, 128))
    w_s = rng.uniform(0.001, 0.02, co).astype(np.float32)
    b = _Builder()
    x = b.tensor("x", [1, h, w_, ci], S.TensorType.INT8, in_s, in_zp)
    if depthwise:
        wq = rng.integers(-127, 128, (1, kh, kw, co)).astype(np.int8)
        wt = b.tensor("w", wq.shape, S.TensorType.INT8, w_s, np.zeros(co, np.int64), axis=3, data=wq)
    else:
        wq = rng.integers(-127, 128, (co, kh, kw, ci)).astype(np.int8)
        wt = b.tensor("w", wq.shape, S.TensorType.INT8, w_s, np.zeros(co, np.int64), axis=0, data=wq)
    bq, b_s = _bias(rng, co, in_s, w_s)
    bt = b.tensor("b", [co], S.TensorType.INT32, b_s, np.zeros(co, np.int64), data=bq)
    oh = -(-h // stride[0]) if padding == "SAME" else (h - kh) // stride[0] + 1
    ow = -(-w_ // stride[1]) if padding == "SAME" else (w_ - kw) // stride[1] + 1
    if oh < 1 or ow < 1:
        return conv2d(rng, depthwise=depthwise, act=act, padding="SAME", stride=stride)
    y = b.tensor("y", [1, oh, ow, co], S.TensorType.INT8, out_s, out_zp)
    if depthwise:
        o = S.DepthwiseConv2DOptionsT()
        o.depthMultiplier = 1
        kind, otype = "DEPTHWISE_CONV_2D", S.BuiltinOptions.DepthwiseConv2DOptions
    else:
        o = S.Conv2DOptionsT()
        kind, otype = "CONV_2D", S.BuiltinOptions.Conv2DOptions
    o.padding, o.strideH, o.strideW = _PADS[padding], stride[0], stride[1]
    o.dilationHFactor = o.dilationWFactor = 1
    o.fusedActivationFunction = _ACTS[act]
    return b.finish(kind, [x, wt, bt], y, otype, o)


def fully_connected(rng, *, per_channel=True, act="NONE"):
    k, co = int(rng.integers(4, 97)), int(rng.integers(2, 33))
    in_s, in_zp = float(rng.uniform(0.005, 0.1)), int(rng.integers(-128, 128))
    out_s, out_zp = float(rng.uniform(0.05, 1.0)), int(rng.integers(-128, 128))
    w_s = rng.uniform(0.001, 0.02, co if per_channel else 1).astype(np.float32)
    b = _Builder()
    x = b.tensor("x", [1, k], S.TensorType.INT8, in_s, in_zp)
    wq = rng.integers(-127, 128, (co, k)).astype(np.int8)
    wt = b.tensor("w", wq.shape, S.TensorType.INT8, w_s, np.zeros(w_s.size, np.int64), data=wq)
    bq, b_s = _bias(rng, co, in_s, w_s)
    bt = b.tensor("b", [co], S.TensorType.INT32, b_s, np.zeros(b_s.size, np.int64), data=bq)
    y = b.tensor("y", [1, co], S.TensorType.INT8, out_s, out_zp)
    o = S.FullyConnectedOptionsT()
    o.fusedActivationFunction = _ACTS[act]
    return b.finish("FULLY_CONNECTED", [x, wt, bt], y, S.BuiltinOptions.FullyConnectedOptions, o)


def average_pool(rng):
    h, w_, c = int(rng.integers(2, 12)), int(rng.integers(2, 12)), int(rng.integers(1, 17))
    kh, kw = int(rng.integers(1, h + 1)), int(rng.integers(1, w_ + 1))
    s, zp = float(rng.uniform(0.01, 0.5)), int(rng.integers(-128, 128))
    b = _Builder()
    x = b.tensor("x", [1, h, w_, c], S.TensorType.INT8, s, zp)
    y = b.tensor("y", [1, h - kh + 1, w_ - kw + 1, c], S.TensorType.INT8, s, zp)
    o = S.Pool2DOptionsT()
    o.padding, o.strideH, o.strideW, o.filterHeight, o.filterWidth = S.Padding.VALID, 1, 1, kh, kw
    o.fusedActivationFunction = S.ActivationFunctionType.NONE
    return b.finish("AVERAGE_POOL_2D", [x], y, S.BuiltinOptions.Pool2DOptions, o)


def softmax(rng):
    n = int(rng.integers(2, 40))
    in_s, in_zp = float(rng.uniform(0.01, 0.5)), int(rng.integers(-128, 128))
    b = _Builder()
    x = b.tensor("x", [1, n], S.TensorType.INT8, in_s, in_zp)
    y = b.tensor("y", [1, n], S.TensorType.INT8, 1.0 / 256, -128)
    o = S.SoftmaxOptionsT()
    o.beta = 1.0
    return b.finish("SOFTMAX", [x], y, S.BuiltinOptions.SoftmaxOptions, o)


def random_model(rng, kind):
    act = str(rng.choice(["NONE", "RELU", "RELU6"]))
    if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        stride = tuple(int(v) for v in rng.integers(1, 3, 2))
        pad = str(rng.choice(["SAME", "VALID"]))
        return conv2d(rng, depthwise=kind == "DEPTHWISE_CONV_2D", act=act, padding=pad, stride=stride)
    if kind == "FULLY_CONNECTED":
        return fully_connected(rng, per_channel=bool(rng.integers(0, 2)), act=act)
    if kind == "AVERAGE_POOL_2D":
        return average_pool(rng)
    if kind == "SOFTMAX":
        return softmax(rng)
    raise ValueError(kind)
