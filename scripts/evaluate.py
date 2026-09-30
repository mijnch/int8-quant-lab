"""FP32 대 INT8 — 정확도·파일 크기·파라미터 수·MAC·x86 지연시간을 잰다.

    python scripts/evaluate.py --saved-model <.../trained_models/kws_ref_model>

측정 대상
- FP32: 레퍼런스 Keras 모델 자체, 그리고 TF 2.21로 변환한 순수 float32 `.tflite`
- MLPerf 배포 `kws_ref_model_float32.tflite`: 이름과 달리 합성곱 가중치가 int8인 동적 범위 모델
- INT8: MLPerf 배포 int8 모델, TF 2.21로 다시 변환한 int8 모델(verify_t1.py가 만든 것)
  × 입력 변환 2가지(레퍼런스 평가 코드의 버림 / 반올림+범위 자르기) × 런타임 3가지

정확도는 tfds speech_commands 테스트 분할 전체(4,890개) top-1이다. MLPerf Tiny 공식 평가
(1,000개 부분집합, 하드웨어 위 측정)와는 다른 자체 측정이다.
"""
import argparse
import platform
import time

from _common import MODELS, REF_FP32, REF_INT8, load_features, save_json

import numpy as np

from qlab import runtime as R
from qlab import spec
from qlab.model import Model
from qlab.stats import mcnemar_exact

RUNTIMES = ("reference", "optimized", "xnnpack")


def cpu_name():
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def latency_ms(model, mode, x1, runs=300, warmup=30):
    it = R.make(model, mode, batch=1)
    for _ in range(warmup):
        R.invoke(it, x1)
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        R.invoke(it, x1)
        t.append(time.perf_counter() - s)
    return float(np.median(t) * 1e3)


def describe(path_or_bytes):
    m = Model(path_or_bytes)
    return {"bytes": m.size_bytes, "params": m.param_count(), "macs": m.macs()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True)
    a = ap.parse_args()

    from qlab import convert
    f = load_features()
    x, y = f["test_x"], f["test_y"]
    km = convert.load_keras(a.saved_model)
    fp32_tfl = convert.to_float(km)
    (MODELS / "kws_tf221_float32.tflite").write_bytes(fp32_tfl)

    res = {"cpu": cpu_name(), "n_test": int(len(y)), "models": {}}
    keras_pred = km.predict(x, batch_size=512, verbose=0)
    res["models"]["fp32_keras"] = {"accuracy": float((keras_pred.argmax(1) == y).mean()),
                                   "params_keras_total": int(km.count_params())}
    for name, model in (("fp32_tflite_tf221", fp32_tfl), ("mlperf_float32_file_dynamic_range", REF_FP32)):
        pred = R.run(model, "optimized", x)
        res["models"][name] = {"accuracy": float((pred.argmax(1) == y).mean()), **describe(model),
                               "latency_ms": {m: latency_ms(model, m, x[:1]) for m in ("optimized", "xnnpack")}}

    preds = {}
    for name, path in (("mlperf_int8", REF_INT8), ("tf221_int8", MODELS / "kws_tf221_int8.tflite")):
        m = Model(path)
        (s, zp), _ = m.io_quant()
        inputs = {"truncate(reference eval)": spec.quantize_input_reference_eval(x, s, zp),
                  "round+clamp": spec.quantize_activation(x, s, zp)}
        v = x / np.float32(s) + zp
        entry = {**describe(path), "input_scale": s, "input_zero_point": zp,
                 "input_values_out_of_int8_range": int(((v < -128) | (v > 127)).sum()),
                 "input_values_total": int(v.size),
                 "input_values_differ_between_methods": int((inputs["truncate(reference eval)"]
                                                             != inputs["round+clamp"]).sum()),
                 "accuracy": {}, "latency_ms": {}}
        for iname, xq in inputs.items():
            for mode in RUNTIMES:
                p = R.run(path, mode, xq)
                preds[(name, iname, mode)] = p
                entry["accuracy"][f"{iname} / {mode}"] = float((p.argmax(1) == y).mean())
        xq1 = inputs["round+clamp"][:1]
        entry["latency_ms"] = {mode: latency_ms(path, mode, xq1) for mode in RUNTIMES}
        agree = {}
        for iname in inputs:
            a_ = [preds[(name, iname, mode)] for mode in RUNTIMES]
            agree[iname] = {
                "samples_output_differs(ref vs xnnpack)": int((a_[0] != a_[2]).any(1).sum()),
                "samples_top1_differs(ref vs xnnpack)": int((a_[0].argmax(1) != a_[2].argmax(1)).sum()),
                "samples_output_differs(ref vs optimized)": int((a_[0] != a_[1]).any(1).sum())}
        entry["runtime_agreement"] = agree
        ok = {k: v.argmax(1) == y for k, v in preds.items() if k[0] == name}
        entry["mcnemar"] = {
            "round+clamp vs truncate (optimized)": mcnemar_exact(
                ok[(name, "round+clamp", "optimized")], ok[(name, "truncate(reference eval)", "optimized")]),
            "xnnpack vs reference (round+clamp)": mcnemar_exact(
                ok[(name, "round+clamp", "xnnpack")], ok[(name, "round+clamp", "reference")])}
        res["models"][name] = entry
    keras_ok = keras_pred.argmax(1) == y
    res["mcnemar_fp32_keras_vs_tf221_int8"] = mcnemar_exact(
        keras_ok, preds[("tf221_int8", "round+clamp", "optimized")].argmax(1) == y)
    save_json("accuracy.json", res)
    for k, v in res["models"].items():
        print(k, v.get("accuracy"))


if __name__ == "__main__":
    main()
