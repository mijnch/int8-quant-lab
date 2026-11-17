"""같은 테스트 세트에서 두 모델의 정확도 차이가 우연 수준인지 본다(McNemar 정확 검정).

두 모델이 모두 맞히거나 모두 틀린 샘플은 차이에 기여하지 않는다. 한쪽만 맞힌 샘플 수
b(A만 정답), c(B만 정답)가 차이를 만든다. 차이가 없다는 가설에서 b는 이항분포 B(b+c, 1/2)를
따르므로, 양측 p값 = min(1, 2·P[X ≤ min(b, c)]).
"""
from math import comb

import numpy as np


def mcnemar_exact(correct_a, correct_b):
    a = np.asarray(correct_a, bool)
    b_ = np.asarray(correct_b, bool)
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    return {"only_a_correct": b, "only_b_correct": c, "p_value": p,
            "accuracy_diff": float(a.mean() - b_.mean())}
