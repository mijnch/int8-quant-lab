"""T1 — 변환기가 정한 양자화 파라미터와 값을 명세 규칙(src/qlab/spec.py)으로 재현한다.

    python scripts/verify_t1.py --saved-model <mlcommons/tiny>/benchmark/training/keyword_spotting/trained_models/kws_ref_model

1) 레퍼런스 Keras 모델을 TF 2.21로 변환한다(레퍼런스와 같은 보정 샘플 120개).
   - models/kws_tf221_int8.tflite        최종 int8 모델
   - models/kws_tf221_calibrated.tflite  양자화 직전 float 모델(텐서별 보정 min/max 기록)
2) 두 파일로 활성값 scale·zero-point, 가중치·bias의 scale과 정수값을 다시 계산해 대조한다.
3) 보정 min/max 자체도 보정용 모델을 직접 돌려 다시 잰다.
4) MLPerf가 배포한 int8 모델(옛 TF로 변환)과 이번 변환 결과가 어디서 같고 다른지 적는다.
"""
import argparse

from _common import MODELS, REF_INT8, load_features, save_json

import numpy as np


def compare_with_reference(ours, ref, cal):
    """연산 순서가 같다는 전제에서, 텐서별 양자화 파라미터와 int8 가중치를 비교한다.
    weight_corr: SavedModel의 (BN을 접은) float 가중치와 MLPerf int8 가중치(역양자화)의 상관계수."""
    from qlab import spec
    rows = []
    for oo, ro, co in zip(ours.ops, ref.ops, cal.ops):
        ot, rt = ours.tensors[oo.outputs[0]], ref.tensors[ro.outputs[0]]
        row = {"op": f"{oo.index}:{oo.kind}",
               "out_scale_equal": bool(ot.scale[0] == rt.scale[0]),
               "out_zp_equal": bool(ot.zero_point[0] == rt.zero_point[0])}
        if oo.kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
            ow, rw = ours.tensors[oo.inputs[1]], ref.tensors[ro.inputs[1]]
            row["weight_scales"] = f"{len(ow.scale)} vs {len(rw.scale)}"
            if ow.data.shape == rw.data.shape and len(ow.scale) == len(rw.scale):
                row["weight_int8_mismatch"] = int((ow.data != rw.data).sum())
                row["weight_scale_max_rel_diff"] = float(np.max(np.abs(ow.scale - rw.scale) / rw.scale))
            deq = spec.dequantize(rw.data, rw.scale, rw.zero_point, axis=rw.axis if len(rw.scale) > 1 else None)
            row["weight_corr_savedmodel_vs_mlperf_int8"] = float(
                np.corrcoef(cal.tensors[co.inputs[1]].data.ravel(), deq.ravel())[0, 1])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True)
    a = ap.parse_args()

    from qlab import convert, t1
    from qlab.model import Model

    cal_x = load_features()["cal_x"]
    km = convert.load_keras(a.saved_model)
    q_bytes = convert.to_int8(km, cal_x)
    c_bytes = convert.to_int8(km, cal_x, calibrate_only=True)
    (MODELS / "kws_tf221_int8.tflite").write_bytes(q_bytes)
    (MODELS / "kws_tf221_calibrated.tflite").write_bytes(c_bytes)

    ours, cal = Model(q_bytes), Model(c_bytes)
    result = t1.verify(cal, ours)
    result["stats_recomputed"] = {b: t1.recompute_stats(c_bytes, cal_x, backend=b) for b in ("tf", "litert")}
    result["summary"]["stats_equal_tf_runtime"] = \
        f'{sum(r["equal"] for r in result["stats_recomputed"]["tf"])}/{len(result["stats_recomputed"]["tf"])}'
    result["summary"]["stats_equal_litert_runtime"] = \
        f'{sum(r["equal"] for r in result["stats_recomputed"]["litert"])}/{len(result["stats_recomputed"]["litert"])}'
    result["vs_mlperf_int8"] = compare_with_reference(ours, Model(REF_INT8), cal)
    result["sizes"] = {"kws_tf221_int8.tflite": len(q_bytes), "mlperf kws_ref_model.tflite": REF_INT8.stat().st_size}
    save_json("t1.json", result)
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
