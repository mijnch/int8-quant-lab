"""이 컴퓨터의 LiteRT가 연산·실행 경로마다 어떤 재양자화 규칙을 쓰는지 실측으로 표를 만든다.

    python scripts/probe_rounding.py [--models 40]

TensorFlow·데이터 없이 돈다. 무작위 단일 연산 모델(src/qlab/synth.py)을 세 실행 경로에서 돌려,
규칙마다 "출력이 전부 같았던 모델 수"를 센다. 현재 정책(model.POLICIES)이 고른 규칙이 모든 모델에서
맞으면 그 칸에 *를 붙인다. 다른 플랫폼(Windows, ARM 등)에서 커널 경로가 다를 때 원인을 좁히는 용도다.
"""
import argparse
import platform

from _common import ROOT  # noqa: F401  (src를 import 경로에 넣는다)

import numpy as np

from qlab import runtime as R
from qlab import synth
from qlab.kernels import ROUNDINGS
from qlab.model import POLICIES, Model

RUNTIMES = ("reference", "optimized", "xnnpack")
KINDS = ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "AVERAGE_POOL_2D", "SOFTMAX")


def variants(kind):
    if kind == "SOFTMAX":
        return "softmax", ("reference", "optimized")
    if kind == "AVERAGE_POOL_2D":
        return None, ("-",)
    return "rounding", ROUNDINGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, default=40, help="연산 종류마다 만들 무작위 모델 수")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(f"{platform.platform()} · {platform.machine()} · models/kind={a.models}")
    rng = np.random.default_rng(a.seed)
    ok = True
    for kind in KINDS:
        key, names = variants(kind)
        count = {m: dict.fromkeys(names, 0) for m in RUNTIMES}
        for _ in range(a.models):
            blob = synth.random_model(rng, kind)
            m = Model(blob)
            shape = m.tensors[m.inputs[0]].shape
            x = rng.integers(-128, 128, size=(32,) + tuple(shape[1:]), dtype=np.int8)
            for mode in RUNTIMES:
                lib = R.invoke(R.make(blob, mode, batch=len(x)), x)
                for v in names:
                    mine = m.run_op(m.ops[0], x, **({key: v} if key else {}), policy=mode)
                    count[mode][v] += int(np.array_equal(mine, lib))
        print(f"\n{kind}")
        print("  runtime    " + "".join(f"{v:>10s}" for v in names))
        for mode in RUNTIMES:
            chosen = POLICIES[mode].get(kind, "-") if key else "-"
            cells = []
            for v in names:
                star = "*" if v == chosen else " "
                cells.append(f"{count[mode][v]:>9d}{star}")
            print(f"  {mode:10s} " + "".join(cells))
            ok &= count[mode][chosen] == a.models
    print("\n현재 정책이 모든 모델에서 일치:", "예" if ok else "아니오")


if __name__ == "__main__":
    main()
