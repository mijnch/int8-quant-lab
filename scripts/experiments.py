"""설계 변수 비교 실험.

    python scripts/experiments.py --saved-model <.../trained_models/kws_ref_model> [--seeds 5]

A. 채널별(per-channel) 대 텐서별(per-tensor) 가중치 — 실제 int8 변환(텐서별은 비공개 옵션
   `_experimental_disable_per_channel`)
B. 보정 샘플 수 1·10·100·500 — 검증 분할에서 무작위로 뽑아 변환, 시드마다 반복
C. 가중치 대칭 대 비대칭 × 텐서별 대 채널별 × 8·4비트 — 시뮬레이션(가중치만 fake quant,
   활성값 float). 변환기는 가중치를 항상 대칭으로만 만들어 실제 int8로는 비교할 수 없다.
D. 층별 민감도 — 가장 크게 무너진 설정(4비트·텐서별·대칭)에서 한 층만 양자화하거나,
   한 층만 float로 남겨 어느 층이 정확도를 좌우하는지 본다.

정확도는 모두 테스트 분할 4,890개 top-1, int8 입력은 반올림+범위 자르기, 런타임은 최적화 커널.
"""
import argparse

from _common import MODELS, load_features, save_json

import numpy as np

from qlab import runtime as R
from qlab import simulate, spec
from qlab.model import Model
from qlab.stats import mcnemar_exact


def int8_correct(blob, x, y):
    m = Model(blob)
    (s, zp), _ = m.io_quant()
    p = R.run(blob, "optimized", spec.quantize_activation(x, s, zp))
    return p.argmax(1) == y


def float_correct(blob, x, y):
    return R.run(blob, "optimized", x).argmax(1) == y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    from qlab import convert
    f = load_features()
    x, y, cal = f["test_x"], f["test_y"], f["cal_x"]
    km = convert.load_keras(a.saved_model)
    out = {}

    # A. per-channel vs per-tensor (보정 세트는 레퍼런스 120개)
    out["granularity"] = {}
    correct = {}
    for name, pc in (("per_channel", True), ("per_tensor", False)):
        blob = convert.to_int8(km, cal, per_channel=pc)
        m = Model(blob)
        correct[name] = int8_correct(blob, x, y)
        out["granularity"][name] = {
            "accuracy": float(correct[name].mean()), "bytes": len(blob),
            "weight_scales_per_layer": [len(m.tensors[op.inputs[1]].scale) for op in m.ops
                                        if op.kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED")]}
        print("A", name, out["granularity"][name]["accuracy"])
    out["granularity"]["mcnemar"] = mcnemar_exact(correct["per_channel"], correct["per_tensor"])

    # B. 보정 샘플 수
    rng = np.random.default_rng(0)
    out["calibration_size"] = {"reference_120": out["granularity"]["per_channel"]["accuracy"]}
    for n in (1, 10, 100, 500):
        accs = []
        for _ in range(a.seeds):
            idx = rng.choice(len(f["val_x"]), size=n, replace=False)
            accs.append(float(int8_correct(convert.to_int8(km, f["val_x"][idx]), x, y).mean()))
        out["calibration_size"][str(n)] = accs
        print("B", n, np.mean(accs), min(accs), max(accs))

    # C. 가중치 양자화 시뮬레이션
    base_path = MODELS / "kws_tf221_float32.tflite"
    if not base_path.exists():
        base_path.write_bytes(convert.to_float(km))
    base = base_path.read_bytes()
    correct["float"] = float_correct(base, x, y)
    out["float_vs_int8"] = mcnemar_exact(correct["float"], correct["per_channel"])
    out["weight_sim"] = {"float_baseline": float(correct["float"].mean()), "configs": []}
    for bits in (8, 4):
        for per_channel in (True, False):
            for symmetric in (True, False):
                blob, sq = simulate.quantize_weights_in_model(base, bits, symmetric, per_channel)
                ok = float_correct(blob, x, y)
                correct[(bits, per_channel, symmetric)] = ok
                row = {"bits": bits, "per_channel": per_channel, "symmetric": symmetric,
                       "accuracy": float(ok.mean()),
                       "mcnemar_vs_float": mcnemar_exact(correct["float"], ok),
                       "sqnr_db_mean": float(np.mean(sq)), "sqnr_db_min": float(np.min(sq)),
                       "sqnr_db_per_layer": [round(v, 2) for v in sq]}
                out["weight_sim"]["configs"].append(row)
                print("C", bits, per_channel, symmetric, row["accuracy"], round(row["sqnr_db_mean"], 2))
    # D. 층별 민감도 (4비트, 텐서별, 대칭)
    layers = [i for i, op in enumerate(Model(base).ops)
              if op.kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED")]
    kinds = {i: op.kind for i, op in enumerate(Model(base).ops)}
    out["layer_sensitivity"] = {"setting": "4-bit, per-tensor, symmetric", "rows": []}
    for i in layers:
        only, _ = simulate.quantize_weights_in_model(base, 4, True, False, ops={i})
        rest, _ = simulate.quantize_weights_in_model(base, 4, True, False, ops=set(layers) - {i})
        row = {"op": f"{i}:{kinds[i]}",
               "accuracy_only_this_layer_quantized": float(float_correct(only, x, y).mean()),
               "accuracy_all_but_this_layer_quantized": float(float_correct(rest, x, y).mean())}
        out["layer_sensitivity"]["rows"].append(row)
        print("D", row)
    out["weight_sim"]["symmetric_vs_asymmetric"] = [
        {"bits": bits, "per_channel": pc,
         **mcnemar_exact(correct[(bits, pc, True)], correct[(bits, pc, False)])}
        for bits in (8, 4) for pc in (True, False)]
    save_json("experiments.json", out)


if __name__ == "__main__":
    main()
