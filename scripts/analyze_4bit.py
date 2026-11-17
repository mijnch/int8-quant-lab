"""4비트 텐서별 대칭 가중치 양자화는 왜 57.4%로 무너졌나 — 가중치 통계, 균일 잡음 모델, 절제 실험.

    python scripts/analyze_4bit.py        # → results/analysis_4bit.json, results/figures/analysis_4bit_*.png

docs/analysis_4bit.md 의 수치는 전부 이 스크립트가 만든다. TensorFlow는 필요 없다 — experiments.py와
같은 float `.tflite`(models/kws_tf221_float32.tflite)의 가중치 버퍼만 바꿔 끼우고(활성값은 float),
LiteRT 최적화 커널로 테스트 분할 4,890개 top-1을 잰다.

1. 층별 가중치 통계 — 채널별 max|w|의 분포(최대/중앙값 비), 텐서별·채널별 간격 Δ,
   |w| < Δ/2 라서 0으로 반올림되는 가중치의 비율, 균일 잡음 모델(Δ²/12)이 예측한 SQNR 대 실측 SQNR.
2. 층 6(1×1 합성곱, 64×64)의 오차 행렬 E = W − Q 의 특이값 — 오차가 저랭크인지.
3. 절제 실험 — 층 6만, 그리고 전 층에서: 이상치 채널 k개를 float로 두기, 간격을 k번째 채널 최대로
   정하고 이상치를 잘라내기(클리핑), 0으로 반올림되는 가중치만 0으로 만들기, 큰 가중치만 격자에 올리기.
"""
import argparse

from _common import MODELS, load_features, save_json, RESULTS

import numpy as np

from qlab import runtime as R
from qlab import simulate
from qlab.fixedpoint import round_half_away
from qlab.model import Model
from qlab.simulate import _WEIGHT_AXIS, fake_quant, sqnr_db
from qlab.stats import mcnemar_exact

BITS = 4
QMAX = 2 ** (BITS - 1) - 1          # 좁은 범위 대칭: ±7
FOCUS = 6                           # 혼자서 가장 크게 떨어뜨린 층(results.md §5-D)


# ------------------------------------------------------------------ 가중치 다루기
def weight_layers(blob):
    """[(op 번호, 종류, float64 가중치)] — CONV·DEPTHWISE·FC 순서대로."""
    m = Model(blob)
    return [(op.index, op.kind, np.asarray(m.tensors[op.inputs[1]].data, np.float64))
            for op in m.ops if op.kind in _WEIGHT_AXIS]


def channel_view(w, kind):
    """(채널 수, 채널당 가중치 수)로 본다. 채널 = 출력 채널(변환기의 채널별 scale 축)."""
    axis = _WEIGHT_AXIS[kind]
    return np.moveaxis(w, axis, 0).reshape(w.shape[axis], -1)


def from_channel_view(wc, w, kind):
    axis = _WEIGHT_AXIS[kind]
    shape = (w.shape[axis],) + tuple(s for i, s in enumerate(w.shape) if i != axis)
    return np.moveaxis(wc.reshape(shape), 0, axis)


def quant_sym(w, scale, qmax=QMAX):
    """대칭 좁은 범위 양자화→역양자화. scale은 스칼라나 w에 브로드캐스트되는 배열."""
    return np.clip(round_half_away(w / scale), -qmax, qmax) * scale


def outlier_order(w, kind):
    """채널별 max|w| 내림차순 채널 번호."""
    return np.argsort(-np.abs(channel_view(w, kind)).max(1))


def keep_outliers_float(w, kind, k):
    """이상치 채널 k개는 float 그대로, 나머지는 나머지의 최대로 정한 텐서별 간격으로 양자화."""
    wc = channel_view(w, kind).copy()
    top = outlier_order(w, kind)[:k]
    rest = np.ones(len(wc), bool)
    rest[top] = False
    scale = np.abs(wc[rest]).max() / QMAX
    wc[rest] = quant_sym(wc[rest], scale)
    return from_channel_view(wc, w, kind)


def clip_to_kth(w, kind, k):
    """간격을 k번째로 큰 채널 최대값으로 정한다(k=1이면 보통의 텐서별). 그보다 큰 가중치는 ±7로 포화."""
    cmax = np.sort(np.abs(channel_view(w, kind)).max(1))[::-1]
    return quant_sym(w, cmax[k - 1] / QMAX)


def zero_small_only(w):
    """0으로 반올림될 가중치(|w| < Δ/2)만 0으로 만들고 나머지는 float 그대로."""
    scale = np.abs(w).max() / QMAX
    return np.where(np.abs(w) < scale / 2, 0.0, w)


def grid_large_only(w):
    """|w| ≥ Δ/2인 가중치만 텐서별 격자에 올리고, 작은 가중치는 float 그대로."""
    scale = np.abs(w).max() / QMAX
    return np.where(np.abs(w) < scale / 2, w, quant_sym(w, scale))


# ------------------------------------------------------------------ 1. 통계
def layer_stats(i, kind, w):
    wc = channel_view(w, kind)
    cmax = np.abs(wc).max(1)
    n = w.size
    scale_t = np.abs(w).max() / QMAX
    scale_c = cmax / QMAX
    power = float(np.mean(w ** 2))
    rows = {}
    for name, bits, per_channel in (("pt4", 4, False), ("pc4", 4, True), ("pt8", 8, False)):
        q = 2 ** (bits - 1) - 1
        s = (cmax / q)[:, None] if per_channel else np.abs(w).max() / q
        wq = quant_sym(wc, s, q)
        err = wc - wq
        small = np.abs(wc) < s / 2                     # 0으로 반올림되는 가중치
        mse = float(np.mean(err ** 2))
        pred = float(np.mean(np.broadcast_to(s, wc.shape) ** 2) / 12)   # 채널별이면 채널 평균
        rows[name] = {
            "bits": bits, "per_channel": per_channel,
            "zero_fraction": float(small.mean()),
            "mse_measured": mse, "mse_uniform_model": pred,
            "mse_from_zeroed": float(np.sum(wc[small] ** 2) / n),
            "mse_from_rest": float(np.sum(err[~small] ** 2) / n),
            "mse_uniform_model_rest_only": float(np.mean(np.broadcast_to(s, wc.shape)[~small] ** 2) / 12)
            if (~small).any() else 0.0,
            "sqnr_db_measured": sqnr_db(wc, wq),
            "sqnr_db_uniform_model": float(10 * np.log10(power / pred)),
        }
    med = float(np.median(cmax))
    return {
        "op": f"{i}:{kind}", "shape": list(w.shape), "n": int(n), "channels": int(len(cmax)),
        "absmax": float(np.abs(w).max()), "channel_absmax_median": med,
        "outlier_ratio_max_over_median": float(cmax.max() / med),
        "channel_absmax_top3": [float(v) for v in np.sort(cmax)[::-1][:3]],
        "scale_per_tensor_4bit": float(scale_t),
        "scale_per_channel_4bit_median": float(np.median(scale_c)),
        "rms": float(np.sqrt(power)),
        "quant": rows,
    }


def error_spectrum(w):
    """1×1 합성곱 W(64×64)와 오차 E = W − Q의 특이값 에너지 — 상위 p개가 차지하는 비율."""
    W = w.reshape(w.shape[0], -1)
    out = {}
    for name, bits in (("W", None), ("E_pt4", 4), ("E_pt8", 8)):
        if bits is None:
            A = W
        else:
            q = 2 ** (bits - 1) - 1
            A = W - quant_sym(W, np.abs(W).max() / q, q)
        out[name] = _spectrum(A)
    # 구조 없는 잡음의 기준선: 같은 크기의 iid 가우스 행렬(시드 20개 평균)
    rng = np.random.default_rng(0)
    base = [_spectrum(rng.standard_normal(W.shape))["top_p_energy_fraction"] for _ in range(20)]
    out["iid_gaussian_baseline"] = {"top_p_energy_fraction": {
        p: float(np.mean([b[p] for b in base])) for p in base[0]}}
    return out


def _spectrum(A):
    s = np.linalg.svd(A, compute_uv=False)
    energy = s ** 2
    cum = np.cumsum(energy) / energy.sum()
    return {"top_p_energy_fraction": {str(p): float(cum[p - 1]) for p in (1, 2, 4, 8, 16, 32)},
            "singular_values_top8": [float(v) for v in s[:8]]}


# ------------------------------------------------------------------ 2. 활성값과 출력 오차
_IN_AXIS = {"CONV_2D": 3, "DEPTHWISE_CONV_2D": 3, "FULLY_CONNECTED": 1}   # 가중치의 입력 채널 축


class _Preserved:
    """중간 텐서를 보존하는 인터프리터를 배치 크기별로 만들어 둔다."""

    def __init__(self, blob):
        self.blob, self.its = blob, {}

    def run(self, chunk, tensor_ids):
        it = self.its.get(len(chunk))
        if it is None:
            it = self.its[len(chunk)] = R.make(self.blob, "optimized", batch=len(chunk), preserve=True)
        R.invoke(it, chunk)
        return {t: it.get_tensor(t).astype(np.float64) for t in tensor_ids}


def activation_analysis(base, x, weights, kinds, batch=512):
    """층마다: 입력 활성값의 채널별 RMS(테스트 분할 전체)와 그 최대/중앙값 비, 그리고 이 층만 4비트
    텐서별로 양자화했을 때 이 층 출력과 최종 logits의 SQNR. 활성값 가중 SQNR(입력 채널 RMS로
    가중한 가중치 오차)이 실측 출력 SQNR을 얼마나 잘 예측하는지 본다."""
    m = Model(base)
    ops = {op.index: op for op in m.ops}
    logits_t = m.outputs[0]
    ids = list(weights)
    in_t = {i: ops[i].inputs[0] for i in ids}
    out_t = {i: ops[i].outputs[0] for i in ids}
    chunks = [x[s:s + batch] for s in range(0, len(x), batch)]

    # 입력 활성값 채널별 제곱합 (채널 = 마지막 축)
    fl = _Preserved(base)
    sq = {i: 0.0 for i in ids}
    cnt = {i: 0 for i in ids}
    float_out = []
    for c in chunks:
        t = fl.run(c, list(in_t.values()) + list(out_t.values()) + [logits_t])
        for i in ids:
            a = t[in_t[i]].reshape(-1, t[in_t[i]].shape[-1])
            sq[i] = sq[i] + (a ** 2).sum(0)
            cnt[i] += len(a)
        float_out.append({**{i: t[out_t[i]] for i in ids}, "logits": t[logits_t]})
    rms = {i: np.sqrt(sq[i] / cnt[i]) for i in ids}

    rows = []
    for i in ids:
        w, kind = weights[i], kinds[i]
        wq = fake_quant(w, BITS, True, None)
        e = w - wq
        s_c = rms[i]
        ax = _IN_AXIS[kind]
        shape = [1] * w.ndim
        shape[ax] = w.shape[ax]
        weight = s_c.reshape(shape)                      # 입력 채널 c의 가중치에 RMS_c를 곱한다
        weighted_sqnr = float(10 * np.log10(np.sum((w * weight) ** 2) / np.sum((e * weight) ** 2)))
        # 이 층만 양자화한 모델의 출력 오차(실측)
        qm = _Preserved(simulate.map_weights(base, lambda _i, _k, ww: fake_quant(ww, BITS, True, None), {i}))
        sig = err = lsig = lerr = 0.0
        for c, fo in zip(chunks, float_out):
            t = qm.run(c, [out_t[i], logits_t])
            sig += np.sum(fo[i] ** 2)
            err += np.sum((t[out_t[i]] - fo[i]) ** 2)
            lsig += np.sum(fo["logits"] ** 2)
            lerr += np.sum((t[logits_t] - fo["logits"]) ** 2)
        rows.append({"op": f"{i}:{kind}",
                     "input_channels": int(len(s_c)),
                     "input_rms_max_over_median": float(s_c.max() / np.median(s_c)),
                     "input_rms_top3_over_median": [float(v / np.median(s_c)) for v in np.sort(s_c)[::-1][:3]],
                     "weight_sqnr_db": sqnr_db(w, wq),
                     "activation_weighted_sqnr_db": weighted_sqnr,
                     "layer_output_sqnr_db": float(10 * np.log10(sig / err)),
                     "logits_sqnr_db": float(10 * np.log10(lsig / lerr))})
        print(f"{rows[-1]['op']:22s} in-rms max/med {rows[-1]['input_rms_max_over_median']:5.2f}  "
              f"wSQNR {rows[-1]['weight_sqnr_db']:5.1f}  act-weighted {weighted_sqnr:5.1f}  "
              f"out {rows[-1]['layer_output_sqnr_db']:5.1f}  logits {rows[-1]['logits_sqnr_db']:5.1f}")
    return rows


# ------------------------------------------------------------------ 3. 절제
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    base = (MODELS / "kws_tf221_float32.tflite").read_bytes()
    f = load_features()
    x, y = f["test_x"], f["test_y"]
    layers = weight_layers(base)
    ids = [i for i, _, _ in layers]
    kinds = {i: k for i, k, _ in layers}
    weights = {i: w for i, _, w in layers}

    out = {"setting": f"{BITS}-bit, per-tensor, symmetric(narrow), weights only (activations float)",
           "layers": [layer_stats(i, k, w) for i, k, w in layers],
           "focus_layer": f"{FOCUS}:{kinds[FOCUS]}",
           "error_spectrum_focus": error_spectrum(weights[FOCUS])}
    for row in out["layers"]:
        print(f"{row['op']:22s} max/med {row['outlier_ratio_max_over_median']:5.2f}  "
              f"zero(pt4) {row['quant']['pt4']['zero_fraction']:.3f}  "
              f"SQNR pt4 {row['quant']['pt4']['sqnr_db_measured']:5.1f} (model {row['quant']['pt4']['sqnr_db_uniform_model']:5.1f})  "
              f"pt8 {row['quant']['pt8']['sqnr_db_measured']:5.1f} (model {row['quant']['pt8']['sqnr_db_uniform_model']:5.1f})")

    def acc(blob):
        return R.run(blob, "optimized", x).argmax(1) == y

    def pt4(_i, _k, w):
        return fake_quant(w, BITS, True, None)

    correct = {"float": acc(base)}
    ablation = {}

    def run(name, fn, ops, ref):
        ok = acc(simulate.map_weights(base, fn, ops))
        correct[name] = ok
        ablation[name] = {"accuracy": float(ok.mean()),
                          "mcnemar_vs_" + ref: mcnemar_exact(ok, correct[ref])}
        print(f"{name:40s} {ok.mean():.4f}")

    # 층 6만 바꾸고 나머지는 float
    run("focus_pt4", pt4, {FOCUS}, "float")
    run("focus_pc4", lambda i, k, w: fake_quant(w, BITS, True, _WEIGHT_AXIS[k]), {FOCUS}, "float")
    for k in (1, 2, 4, 8):
        run(f"focus_pt4_keep_top{k}_float", lambda i, kd, w, k=k: keep_outliers_float(w, kd, k), {FOCUS}, "focus_pt4")
    for k in (2, 3, 5, 9):
        run(f"focus_pt4_clip_to_{k}th_channel", lambda i, kd, w, k=k: clip_to_kth(w, kd, k), {FOCUS}, "focus_pt4")
    run("focus_zero_small_only", lambda i, k, w: zero_small_only(w), {FOCUS}, "focus_pt4")
    run("focus_grid_large_only", lambda i, k, w: grid_large_only(w), {FOCUS}, "focus_pt4")

    # 전 층
    all_ops = set(ids)
    run("all_pt4", pt4, all_ops, "float")
    run("all_pc4", lambda i, k, w: fake_quant(w, BITS, True, _WEIGHT_AXIS[k]), all_ops, "float")
    for k in (1, 2, 4, 8):
        run(f"all_pt4_keep_top{k}_float", lambda i, kd, w, k=k: keep_outliers_float(w, kd, k), all_ops, "all_pt4")
    for k in (2, 3, 5):
        run(f"all_pt4_clip_to_{k}th_channel", lambda i, kd, w, k=k: clip_to_kth(w, kd, k), all_ops, "all_pt4")
    run("all_zero_small_only", lambda i, k, w: zero_small_only(w), all_ops, "all_pt4")
    run("all_grid_large_only", lambda i, k, w: grid_large_only(w), all_ops, "all_pt4")
    out["float_accuracy"] = float(correct["float"].mean())
    out["ablation"] = ablation
    out["activation"] = activation_analysis(base, x, weights, kinds)

    # 이상치 채널을 float로 둘 때 전체 가중치 중 float로 남는 비율(비용)
    out["keep_top_float_cost"] = {
        str(k): float(sum(k * channel_view(w, kinds[i]).shape[1] for i, w in weights.items())
                      / sum(w.size for w in weights.values())) for k in (1, 2, 4, 8)}
    save_json("analysis_4bit.json", out)
    if not a.no_figures:
        figures(out, weights, kinds)


def figures(out, weights, kinds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(exist_ok=True)

    # (a) 층별 채널 max|w| 분포 — 점 하나가 채널 하나, 가로선은 텐서별 Δ/2 (이 아래 가중치는 0이 된다)
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (i, w) in enumerate(weights.items()):
        cmax = np.abs(channel_view(w, kinds[i])).max(1)
        ax.scatter(np.full(len(cmax), j) + np.random.default_rng(0).uniform(-0.2, 0.2, len(cmax)),
                   cmax, s=8, alpha=0.6, color="C0")
        ax.hlines(np.abs(w).max() / QMAX / 2, j - 0.35, j + 0.35, color="C3", lw=1.5)
    ax.set_xticks(range(len(weights)), [f"{i}:{kinds[i][:4]}" for i in weights], rotation=30)
    ax.set_yscale("log")
    ax.set_ylabel("per-channel max|w| (dots), per-tensor 4-bit Δ/2 (red)")
    ax.set_title("Per-channel max|w| by layer — the largest channel sets Δ for the whole tensor")
    fig.tight_layout()
    fig.savefig(fig_dir / "analysis_4bit_channels.png", dpi=150)

    # (b) 층 6: |w| 히스토그램과 Δ/2
    w = weights[FOCUS]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(np.abs(w).ravel(), bins=120, color="C0")
    for bits, c in ((4, "C3"), (8, "C2")):
        half = np.abs(w).max() / (2 ** (bits - 1) - 1) / 2
        ax.axvline(half, color=c, ls="--", label=f"per-tensor {bits}-bit Δ/2 = {half:.4f}")
    ax.set_xlabel("|w| of layer 6 (1×1 conv, 64×64)")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title(f"Layer {FOCUS}: weights left of the red line round to zero at 4 bits")
    fig.tight_layout()
    fig.savefig(fig_dir / "analysis_4bit_hist.png", dpi=150)

    # (c) 절제 — 이상치 채널 k개를 float로 둘 때 정확도
    ab = out["ablation"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ks = [0, 1, 2, 4, 8]
    for prefix, label in (("focus", f"layer {FOCUS} only"), ("all", "all layers")):
        ys = [ab[f"{prefix}_pt4"]["accuracy"]] + [ab[f"{prefix}_pt4_keep_top{k}_float"]["accuracy"] for k in ks[1:]]
        ax.plot(ks, [100 * v for v in ys], marker="o", label=label)
        ax.axhline(100 * ab[f"{prefix}_pc4"]["accuracy"], ls=":", color=ax.lines[-1].get_color(),
                   label=f"{label}, per-channel 4-bit")
    ax.axhline(100 * out["float_accuracy"], color="k", lw=0.8, label="float")
    ax.set_xlabel("outlier channels kept in float (per layer)")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.legend(fontsize=8)
    ax.set_title("4-bit per-tensor: accuracy vs number of outlier channels left in float")
    fig.tight_layout()
    fig.savefig(fig_dir / "analysis_4bit_ablation.png", dpi=150)
    print("→ results/figures/analysis_4bit_*.png")


if __name__ == "__main__":
    main()
