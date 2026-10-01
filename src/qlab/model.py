"""`.tflite` 파일을 직접 읽어(인터프리터 없이) 연산 목록·가중치·양자화 파라미터를 꺼내고,
kernels.py로 망 전체를 정수 산술로 실행한다.

flatbuffer 스키마는 ai-edge-litert에 포함된 `schema_py_generated`를 쓴다.
"""
import sys
from dataclasses import dataclass, field

import numpy as np
from ai_edge_litert import schema_py_generated as schema

from . import kernels as K

_OP_NAMES = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
_ACT = {v: k for k, v in vars(schema.ActivationFunctionType).items() if not k.startswith("_")}
_PAD = {schema.Padding.SAME: "SAME", schema.Padding.VALID: "VALID"}
_DTYPES = {schema.TensorType.INT8: np.int8, schema.TensorType.INT32: np.int32,
           schema.TensorType.FLOAT32: np.float32, schema.TensorType.INT64: np.int64}


@dataclass
class Tensor:
    index: int
    name: str
    shape: tuple
    dtype: type
    scale: np.ndarray = None          # float32, 채널 수만큼 또는 1개
    zero_point: np.ndarray = None
    axis: int = 0
    data: np.ndarray = None           # 상수(가중치·bias)일 때만
    stat_min: float = None            # 보정만 한 float 모델(calibrate_only)에 기록된 min/max
    stat_max: float = None


@dataclass
class Op:
    index: int
    kind: str
    inputs: list
    outputs: list
    options: dict = field(default_factory=dict)


# 런타임별로 원문에서 확인한 재양자화 방식(자세한 근거는 docs/notes.md).
POLICIES = {
    # BUILTIN_REF: reference_integer_ops. FC는 double 배율 곱(LiteRT 2.x 참조 커널)
    "reference": {"CONV_2D": "double", "DEPTHWISE_CONV_2D": "double",
                  "FULLY_CONNECTED": "float", "SOFTMAX": "reference"},
    # 최적화 커널(x86): CONV·FC는 cpu_backend_gemm → ruy, DEPTHWISE는 자체 누산 후
    # optimized_ops::Quantize(8채널 묶음은 NEON→SSE 경로, 나머지는 스칼라)
    "optimized": {"CONV_2D": "ruy", "DEPTHWISE_CONV_2D": "neon8",
                  "FULLY_CONNECTED": "ruy", "SOFTMAX": "optimized"},
    # XNNPACK 위임: qs8 fp32 재양자화. SOFTMAX는 위임되지 않아 TFLite 최적화 커널이 돈다
    "xnnpack": {"CONV_2D": "fp32", "DEPTHWISE_CONV_2D": "fp32",
                "FULLY_CONNECTED": "fp32", "SOFTMAX": "optimized"},
}

# Windows 휠(MSVC 빌드)은 최적화 커널의 경로가 다르다(scripts/probe_rounding.py로 CI에서 확인).
# - CONV·FC: ruy의 x86 SIMD 커널은 __AVX2__·__AVX512F__가 컴파일 시점에 정의돼야 들어가는데
#   (ruy platform.h) MSVC는 /arch 없이는 정의하지 않아, 표준 C++ 경로(apply_multiplier.cc,
#   한 번 반올림)를 쓴다.
# - DEPTHWISE: x86의 NEON→SSE 경로는 `__GNUC__ && __SSE4_1__`일 때만 켜져(neon_check.h)
#   MSVC에서는 스칼라 경로(double)만 쓴다.
if sys.platform == "win32":
    POLICIES["optimized"].update({"CONV_2D": "single", "DEPTHWISE_CONV_2D": "double",
                                  "FULLY_CONNECTED": "single"})


class Model:
    def __init__(self, path_or_bytes):
        buf = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) \
            else open(path_or_bytes, "rb").read()
        self.size_bytes = len(buf)
        m = schema.ModelT.InitFromPackedBuf(buf, 0)
        if len(m.subgraphs) != 1:
            raise NotImplementedError("서브그래프 1개인 모델만 다룬다")
        sg = m.subgraphs[0]
        self.tensors = [self._tensor(i, t, m.buffers) for i, t in enumerate(sg.tensors)]
        self.ops = []
        for i, op in enumerate(sg.operators):
            oc = m.operatorCodes[op.opcodeIndex]
            kind = _OP_NAMES[max(oc.builtinCode, oc.deprecatedBuiltinCode)]
            opts = dict(vars(op.builtinOptions)) if op.builtinOptions is not None else {}
            self.ops.append(Op(i, kind, [int(x) for x in op.inputs],
                               [int(x) for x in op.outputs], opts))
        self.inputs = [int(x) for x in sg.inputs]
        self.outputs = [int(x) for x in sg.outputs]

    @staticmethod
    def _tensor(i, t, buffers):
        dtype = _DTYPES.get(t.type)
        shape = tuple(int(s) for s in t.shape) if t.shape is not None else ()
        out = Tensor(i, t.name.decode(), shape, dtype)
        q = t.quantization
        if q is not None and q.scale is not None and len(q.scale):
            out.scale = np.asarray(q.scale, np.float32)
            out.zero_point = np.asarray(q.zeroPoint, np.int64)
            out.axis = int(q.quantizedDimension)
        if q is not None and q.min is not None and len(q.min) == 1:
            out.stat_min, out.stat_max = float(q.min[0]), float(q.max[0])
        b = buffers[t.buffer] if t.buffer < len(buffers) else None
        if b is not None and b.data is not None and len(b.data) and dtype is not None:
            out.data = np.frombuffer(bytes(bytearray(b.data)), dtype=dtype).reshape(shape)
        return out

    # ------------------------------------------------------------------ 요약 지표
    def param_count(self):
        """가중치·bias 원소 수(CONV·DEPTHWISE·FC의 2·3번째 입력). BN은 변환 때 접혀 들어가 있다."""
        idx = set()
        for op in self.ops:
            if op.kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
                idx.update(i for i in op.inputs[1:3] if i >= 0)
        return int(sum(self.tensors[i].data.size for i in idx))

    def macs(self):
        """곱셈-누산 횟수(배치 1). CONV·DEPTHWISE·FC만 센다."""
        total = 0
        for op in self.ops:
            if op.kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
                w = self.tensors[op.inputs[1]]
                out = self.tensors[op.outputs[0]]
                n_out = int(np.prod(out.shape[1:]))
                if op.kind == "CONV_2D":
                    total += n_out * int(np.prod(w.shape[1:]))
                elif op.kind == "DEPTHWISE_CONV_2D":
                    total += n_out * w.shape[1] * w.shape[2]
                else:
                    total += n_out * w.shape[1]
        return total

    # ------------------------------------------------------------------ 연산별 준비
    def op_params(self, op):
        """연산 하나를 실행하는 데 필요한 정수 파라미터(재양자화 곱셈자 등)를 계산한다."""
        t = self.tensors
        x, y = t[op.inputs[0]], t[op.outputs[0]]
        o = op.options
        p = {}
        if op.kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
            w = t[op.inputs[1]]
            b = t[op.inputs[2]] if len(op.inputs) > 2 and op.inputs[2] >= 0 else None
            rq = K.conv_multipliers(x.scale[0], w.scale, y.scale[0])
            if len(w.scale) == 1:
                rq = rq.repeat(w.shape[0] if op.kind != "DEPTHWISE_CONV_2D" else w.shape[3])
            p.update(w=w.data, b=None if b is None else b.data, in_zp=int(x.zero_point[0]),
                     out_zp=int(y.zero_point[0]), rq=rq,
                     act_range=K.activation_range(_ACT[o["fusedActivationFunction"]],
                                                  y.scale[0], int(y.zero_point[0])))
            if op.kind != "FULLY_CONNECTED":
                p.update(stride=(o["strideH"], o["strideW"]), padding=_PAD[o["padding"]],
                         dilation=(o["dilationHFactor"], o["dilationWFactor"]))
            if op.kind == "DEPTHWISE_CONV_2D":
                p["depth_multiplier"] = o["depthMultiplier"]
        elif op.kind == "AVERAGE_POOL_2D":
            if not (x.scale[0] == y.scale[0] and x.zero_point[0] == y.zero_point[0]):
                raise NotImplementedError("int8 AVERAGE_POOL_2D는 입·출력 양자화가 같아야 한다")
            p.update(filter_hw=(o["filterHeight"], o["filterWidth"]),
                     stride=(o["strideH"], o["strideW"]), padding=_PAD[o["padding"]],
                     act_range=K.activation_range(_ACT[o["fusedActivationFunction"]],
                                                  y.scale[0], int(y.zero_point[0])))
        elif op.kind == "SOFTMAX":
            p.update(input_scale=x.scale[0], beta=o.get("beta", 1.0))
        return p

    def run_op(self, op, x, rounding=None, softmax=None, policy="reference"):
        """연산 하나를 numpy로 실행. x는 첫 입력 텐서 값(int8, 배치 차원 포함).
        rounding·softmax를 주지 않으면 policy(POLICIES의 키)에 따른다."""
        pol = POLICIES[policy]
        rounding = rounding or pol.get(op.kind, "double")
        softmax = softmax or pol["SOFTMAX"]
        p = self.op_params(op)
        if op.kind == "CONV_2D":
            return K.conv2d(x, rounding=rounding, **p)
        if op.kind == "DEPTHWISE_CONV_2D":
            return K.depthwise_conv2d(x, rounding=rounding, **p)
        if op.kind == "FULLY_CONNECTED":
            return K.fully_connected(x.reshape(x.shape[0], -1), rounding=rounding,
                                     **{k: v for k, v in p.items()
                                        if k not in ("stride", "padding", "dilation")})
        if op.kind == "AVERAGE_POOL_2D":
            return K.average_pool(x, **p)
        if op.kind == "RESHAPE":
            out_shape = self.tensors[op.outputs[0]].shape
            return x.reshape((x.shape[0],) + tuple(out_shape[1:]))
        if op.kind == "SOFTMAX":
            fn = K.softmax_reference if softmax == "reference" else K.softmax_optimized
            return fn(x, **p)
        raise NotImplementedError(op.kind)

    def run(self, x, policy="reference", keep=False, chunk=512):
        """망 전체를 int8 정수 산술로 실행. keep=True면 모든 중간 텐서를 돌려준다.
        메모리를 아끼려고 chunk개씩 나눠 계산한다(결과는 나누지 않은 것과 같다)."""
        if not keep and len(x) > chunk:
            return np.concatenate([self.run(x[s:s + chunk], policy) for s in range(0, len(x), chunk)])
        values = {self.inputs[0]: x}
        for op in self.ops:
            values[op.outputs[0]] = self.run_op(op, values[op.inputs[0]], policy=policy)
        return values if keep else values[self.outputs[0]]

    def io_quant(self):
        i, o = self.tensors[self.inputs[0]], self.tensors[self.outputs[0]]
        return (float(i.scale[0]), int(i.zero_point[0])), (float(o.scale[0]), int(o.zero_point[0]))
