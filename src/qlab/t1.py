"""T1 대조 — 변환기가 만든 양자화 파라미터·값을 spec.py 규칙으로 다시 계산해 비교한다.

입력은 같은 변환에서 나온 두 파일이다.
- 보정만 한 float 모델(calibrate_only): 접힌(BN 흡수) float 가중치·bias와 텐서별 보정 min/max
- 최종 int8 모델: 변환기가 정한 scale·zero-point와 int8/int32 값

TensorFlow 없이 numpy와 flatbuffer 읽기만으로 돈다(테스트에서 그대로 쓴다).
"""
import numpy as np

from . import spec
from .model import Model

_LINEAR = ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED")


def _activation_expectation(q, op_index, cal_tensor):
    """활성 텐서 하나의 기대 (scale, zp, 규칙 이름)."""
    if op_index is None:                                   # 모델 입력
        s, z = spec.activation_qparams(cal_tensor.stat_min, cal_tensor.stat_max)
        return s, z, "보정 min/max"
    op = q.ops[op_index]
    if op.kind == "SOFTMAX":                               # int8 softmax 출력은 고정
        return np.float32(1 / 256), -128, "softmax 고정값"
    if op.kind in ("AVERAGE_POOL_2D", "RESHAPE"):          # 입·출력 같은 scale을 요구하는 연산
        t = q.tensors[op.inputs[0]]
        return t.scale[0], int(t.zero_point[0]), "입력과 같게"
    s, z = spec.activation_qparams(cal_tensor.stat_min, cal_tensor.stat_max)
    return s, z, "보정 min/max"


def verify(cal: Model, q: Model):
    if [o.kind for o in cal.ops] != [o.kind for o in q.ops]:
        raise ValueError("두 모델의 연산 순서가 다르다")
    out = {"activations": [], "weights": [], "bias": []}

    pairs = [(None, cal.inputs[0], q.inputs[0])] + \
            [(i, co.outputs[0], qo.outputs[0]) for i, (co, qo) in enumerate(zip(cal.ops, q.ops))]
    for op_index, ci, qi in pairs:
        ct, qt = cal.tensors[ci], q.tensors[qi]
        s, z, rule = _activation_expectation(q, op_index, ct)
        out["activations"].append({
            "tensor": qt.name[:60], "op": "INPUT" if op_index is None else q.ops[op_index].kind,
            "rule": rule, "calib_min": ct.stat_min, "calib_max": ct.stat_max,
            "scale_converter": float(qt.scale[0]), "scale_mine": float(s),
            "zp_converter": int(qt.zero_point[0]), "zp_mine": int(z),
            "equal": bool(np.float32(s) == qt.scale[0] and z == qt.zero_point[0])})

    for co, qo in zip(cal.ops, q.ops):
        if qo.kind not in _LINEAR:
            continue
        wf, wq = cal.tensors[co.inputs[1]], q.tensors[qo.inputs[1]]
        axis = wq.axis if len(wq.scale) > 1 else None
        mine_q, mine_s = spec.quantize_weights(wf.data, axis)
        div_q, _ = spec.quantize_weights(wf.data, axis, rule="divide")
        diff = np.abs(mine_q.astype(np.int32) - wq.data.astype(np.int32))
        out["weights"].append({
            "op": f"{qo.index}:{qo.kind}", "shape": list(wq.shape), "per_channel": axis is not None,
            "axis": axis, "scales_equal": bool(np.array_equal(mine_s, wq.scale)),
            "zero_points_all_zero": bool(not wq.zero_point.any()),
            "int8_mismatch": int((diff != 0).sum()), "int8_max_diff": int(diff.max()),
            "int8_mismatch_divide_rule": int((div_q != wq.data).sum()),
            "n": int(wq.data.size)})

        bf, bq = cal.tensors[co.inputs[2]], q.tensors[qo.inputs[2]]
        in_scale = q.tensors[qo.inputs[0]].scale[0]
        mine_bq, mine_bs = spec.quantize_bias(bf.data, in_scale, wq.scale)
        div_bq, _ = spec.quantize_bias(bf.data, in_scale, wq.scale, rule="divide")
        bdiff = np.abs(mine_bq.astype(np.int64) - bq.data.astype(np.int64))
        out["bias"].append({
            "op": f"{qo.index}:{qo.kind}", "n": int(bq.data.size),
            "scales_equal": bool(np.array_equal(mine_bs, bq.scale)),
            "int32_mismatch": int((bdiff != 0).sum()), "int32_max_diff": int(bdiff.max()),
            "int32_mismatch_divide_rule": int((div_bq != bq.data).sum())})

    out["summary"] = {
        "activation_qparams_equal": f'{sum(a["equal"] for a in out["activations"])}/{len(out["activations"])}',
        "weight_scales_equal": f'{sum(w["scales_equal"] for w in out["weights"])}/{len(out["weights"])}',
        "weight_int8_mismatch": sum(w["int8_mismatch"] for w in out["weights"]),
        "weight_count": sum(w["n"] for w in out["weights"]),
        "bias_scales_equal": f'{sum(b["scales_equal"] for b in out["bias"])}/{len(out["bias"])}',
        "bias_int32_mismatch": sum(b["int32_mismatch"] for b in out["bias"]),
        "weight_int8_mismatch_divide_rule": sum(w["int8_mismatch_divide_rule"] for w in out["weights"]),
        "bias_int32_mismatch_divide_rule": sum(b["int32_mismatch_divide_rule"] for b in out["bias"]),
        "bias_count": sum(b["n"] for b in out["bias"])}
    return out


def recompute_stats(cal_bytes, calibration, backend="tf"):
    """보정 min/max를 직접 다시 잰다: 보정용 float 모델을 샘플 1개씩 돌려 활성 텐서마다
    최솟값·최댓값을 모은다(TFLite calibration_logger의 MinMax::Update와 같은 누적).

    backend="tf"    : TensorFlow에 내장된 TFLite 인터프리터(최적화 커널, 위임 없음) — 변환기의
                      보정기(CalibrationWrapper)가 쓰는 것과 같은 바이너리
    backend="litert": ai-edge-litert 인터프리터(최적화 커널, 위임 없음)
    """
    cal = Model(cal_bytes)
    idx = [cal.inputs[0]] + [op.outputs[0] for op in cal.ops]
    if backend == "tf":
        import tensorflow as tf
        it = tf.lite.Interpreter(
            model_content=cal_bytes, num_threads=1, experimental_preserve_all_tensors=True,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        it.allocate_tensors()
    else:
        from . import runtime as R
        it = R.make(cal_bytes, "optimized", batch=1, preserve=True)
    in_index = it.get_input_details()[0]["index"]
    lo = {i: np.float32(np.inf) for i in idx}
    hi = {i: np.float32(-np.inf) for i in idx}
    for x in np.asarray(calibration, np.float32):
        it.set_tensor(in_index, x[None])
        it.invoke()
        for i in idx:
            v = it.get_tensor(i)
            lo[i], hi[i] = min(lo[i], v.min()), max(hi[i], v.max())
    rows = []
    for i in idx:
        t = cal.tensors[i]
        rows.append({"tensor": t.name[:60], "min_logged": t.stat_min, "min_mine": float(lo[i]),
                     "max_logged": t.stat_max, "max_mine": float(hi[i]),
                     "equal": bool(lo[i] == np.float32(t.stat_min) and hi[i] == np.float32(t.stat_max))})
    return rows
