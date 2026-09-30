"""T2·T3 — numpy 정수 커널이 LiteRT의 세 실행 경로와 비트 단위로 같은지 확인한다.

    python scripts/verify_bitexact.py [--fuzz 100]

1) 층별(layer-isolated): 인터프리터가 계산한 층 입력을 그대로 받아 그 층 하나만 numpy로 계산해
   라이브러리 출력과 비교한다. 재양자화 방식 5가지를 모두 돌려 어느 것이 맞는지 표로 남긴다.
   XNNPACK은 노드 융합을 꺼서 연산마다 따로 위임되게 해야 층 출력이 보인다.
2) 망 전체(end-to-end): 테스트 세트 4,890개를 numpy로 처음부터 끝까지 계산해 출력 12개를
   런타임별로 비교한다(런타임마다 원문에서 확인한 방식 = model.POLICIES).
3) 무작위 단일 연산 모델: scale·zero-point·활성함수·보폭·패딩을 바꾼 모델을 만들어 비교한다.
   실제 모델에서는 ReLU가 음수 결과를 잘라내 드러나지 않는 차이까지 가려내기 위해서다.
"""
import argparse
import collections
import platform

from _common import MODELS, REF_INT8, load_features, save_json

import numpy as np

from qlab import runtime as R
from qlab import spec, synth
from qlab.kernels import EXPF_SOURCE, ROUNDINGS
from qlab.model import POLICIES, Model

RUNTIMES = ("reference", "optimized", "xnnpack")


def variants(kind):
    if kind == "SOFTMAX":
        return [("softmax", v) for v in ("reference", "optimized")]
    if kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
        return [("rounding", v) for v in ROUNDINGS]
    return [(None, "-")]


def layer_isolated(model_path, x, mode):
    m = Model(model_path)
    it = R.make(model_path, mode, batch=len(x), preserve=True, split_delegate=(mode == "xnnpack"))
    R.invoke(it, x)
    delegated = "DELEGATE" in R.execution_plan(it)
    rows = []
    for op in m.ops:
        xin, lib = it.get_tensor(op.inputs[0]), it.get_tensor(op.outputs[0])
        res = {}
        for key, v in variants(op.kind):
            mine = m.run_op(op, xin, **({key: v} if key else {}), policy=mode)
            res[v] = int((mine != lib).sum())
        rows.append({"op": f"{op.index}:{op.kind}", "n": int(lib.size), "mismatch": res,
                     "policy": POLICIES[mode].get(op.kind, "-")})
    return {"delegated": delegated, "ops": rows}


def end_to_end(model_path, xq):
    m = Model(model_path)
    out = {}
    for mode in RUNTIMES:
        lib = R.run(model_path, mode, xq)
        mine = m.run(xq, policy=mode)
        out[mode] = {"outputs": int(lib.size), "mismatch": int((mine != lib).sum()),
                     "max_abs_diff": int(np.abs(mine.astype(int) - lib.astype(int)).max())}
    return out


def fuzz(n_per_kind, seed=0):
    rng = np.random.default_rng(seed)
    kinds = ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "AVERAGE_POOL_2D", "SOFTMAX")
    out = {}
    for kind in kinds:
        full = {mode: collections.Counter() for mode in RUNTIMES}
        delegated = 0
        for _ in range(n_per_kind):
            blob = synth.random_model(rng, kind)
            m = Model(blob)
            op = m.ops[0]
            shape = m.tensors[m.inputs[0]].shape
            x = rng.integers(-128, 128, size=(64,) + tuple(shape[1:]), dtype=np.int8)
            for mode in RUNTIMES:
                it = R.make(blob, mode, batch=len(x))
                lib = R.invoke(it, x)
                if mode == "xnnpack":
                    delegated += "DELEGATE" in R.execution_plan(it)
                for key, v in variants(kind):
                    mine = m.run_op(op, x, **({key: v} if key else {}), policy=mode)
                    full[mode][v] += int(np.array_equal(mine, lib))
        out[kind] = {"models": n_per_kind, "xnnpack_delegated": delegated,
                     "all_outputs_equal": {mode: dict(c) for mode, c in full.items()},
                     "policy": {mode: POLICIES[mode].get(kind, "-") for mode in RUNTIMES}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=100, help="연산 종류마다 만들 무작위 모델 수")
    a = ap.parse_args()

    feats = load_features()
    rng = np.random.default_rng(0)
    rand = rng.integers(-128, 128, size=(300, 49, 10, 1), dtype=np.int8)
    models = {"mlperf_int8": REF_INT8, "tf221_int8": MODELS / "kws_tf221_int8.tflite"}

    result = {"environment": {"machine": platform.machine(), "python": platform.python_version(),
                              "expf": EXPF_SOURCE}}
    for name, path in models.items():
        m = Model(path)
        (s, zp), _ = m.io_quant()
        real = spec.quantize_activation(feats["test_x"][:300], s, zp)
        result[name] = {
            "layer_isolated": {f"{mode}/{src}": layer_isolated(path, x, mode)
                               for mode in RUNTIMES for src, x in (("random", rand), ("test", real))},
            "end_to_end_test_set": end_to_end(path, spec.quantize_activation(feats["test_x"], s, zp)),
        }
        print(name, {k: v["mismatch"] for k, v in result[name]["end_to_end_test_set"].items()})
    result["fuzz"] = fuzz(a.fuzz)
    for kind, r in result["fuzz"].items():
        print(kind, {mode: r["all_outputs_equal"][mode].get(r["policy"][mode]) for mode in RUNTIMES},
              "/", r["models"])
    save_json("bitexact.json", result)


if __name__ == "__main__":
    main()
