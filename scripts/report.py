"""results/*.json → results/results.md (표) + results/figures/*.png (그림).

    python scripts/report.py
"""
import importlib.metadata as md
import platform

from _common import RESULTS, load_json

import numpy as np


def pct(v):
    return f"{100 * v:.2f}%"


def p_fmt(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def mc(m, flip=False):
    """McNemar 결과 한 줄. flip=True면 A·B를 바꿔 (B − A)로 적는다."""
    d, a, b = m["accuracy_diff"], m["only_a_correct"], m["only_b_correct"]
    if flip:
        d, a, b = -d, b, a
    return f"{d * 100:+.2f}%p (앞쪽만 정답 {a}개 · 뒤쪽만 정답 {b}개, {p_fmt(m['p_value'])})"


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "-"


def figures(exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(exist_ok=True)

    cs = exp["calibration_size"]
    ns = [k for k in cs if k.isdigit()]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for n in ns:
        ax.scatter([int(n)] * len(cs[n]), [100 * v for v in cs[n]], color="tab:blue", alpha=0.5, s=18)
    ax.plot([int(n) for n in ns], [100 * np.mean(cs[n]) for n in ns], "o-", color="tab:blue",
            label="random subset (mean of seeds)")
    ax.axhline(100 * cs["reference_120"], color="tab:orange", ls="--", label="MLPerf calibration set (120)")
    ax.axhline(100 * exp["weight_sim"]["float_baseline"], color="gray", ls=":", label="FP32 TFLite")
    ax.set_xscale("log")
    ax.set_xlabel("calibration samples")
    ax.set_ylabel("test top-1 accuracy (%)")
    ax.set_title("INT8 accuracy vs. calibration set size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_size.png", dpi=150)
    plt.close(fig)

    rows = exp["weight_sim"]["configs"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=False)
    for ax, bits in zip(axes, (8, 4)):
        sub = [r for r in rows if r["bits"] == bits]
        labels = [f"{'ch' if r['per_channel'] else 'tensor'}\n{'sym' if r['symmetric'] else 'asym'}" for r in sub]
        ax.bar(labels, [100 * r["accuracy"] for r in sub],
               color=["tab:blue" if r["symmetric"] else "tab:green" for r in sub])
        ax.axhline(100 * exp["weight_sim"]["float_baseline"], color="gray", ls=":", label="FP32")
        for i, r in enumerate(sub):
            ax.text(i, 100 * r["accuracy"] + 1, f"{100 * r['accuracy']:.1f}", ha="center", fontsize=8)
        ax.set_ylim(0, 105)          # 0부터 — 8비트의 작은 차이를 부풀려 보이지 않게
        ax.set_title(f"{bits}-bit weights")
        ax.set_ylabel("test top-1 accuracy (%)")
    fig.suptitle("Weight-only fake quantization (activations stay float); dotted line = FP32", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "weight_quant_sim.png", dpi=150)
    plt.close(fig)


def main():
    dc, t1, be, acc, exp = (load_json(n) for n in
                            ("data_check.json", "t1.json", "bitexact.json", "accuracy.json", "experiments.json"))
    figures(exp)
    L = []
    w = L.append
    w("# 결과\n")
    w("> `scripts/report.py`가 `results/*.json`에서 자동으로 만든 문서입니다. 수치를 고치려면 JSON을 만든 "
      "스크립트를 다시 실행하세요.\n")
    w("## 측정 환경\n")
    w(f"- CPU: {acc['cpu']} (클라우드 x86 VM, 스레드 1개)")
    w(f"- Python {platform.python_version()}, TensorFlow {version('tensorflow')}, tf-keras {version('tf-keras')}, "
      f"ai-edge-litert {version('ai-edge-litert')}, numpy {version('numpy')}")
    w(f"- 데이터: tfds `speech_commands` 0.0.3 — 테스트 {dc['test']['n']:,}개, 검증 {dc['val']['n']:,}개, "
      f"보정 {dc['cal']['n']}개(MLPerf `quant_cal_idxs.txt`)\n")

    w("## 1. 전처리 대조 — 이 저장소의 MFCC 구현 대 MLPerf 레퍼런스 코드\n")
    w("| 분할 | 샘플 수 | 특징값 비트 일치 | 레이블 일치 |\n|---|---:|:---:|:---:|")
    for k, name in (("test", "테스트"), ("val", "검증"), ("cal", "보정")):
        d = dc[k]
        w(f"| {name} | {d['n']:,} | {'예' if d['features_bit_equal'] else '아니오'} | {'예' if d['labels_equal'] else '아니오'} |")
    w("")

    s = t1["summary"]
    w("## 2. T1 — 변환기가 정한 양자화 파라미터·값의 재현\n")
    w("대상: 레퍼런스 Keras 모델을 TF 2.21로 변환한 `models/kws_tf221_int8.tflite`. "
      "재현 입력: 같은 변환의 보정 모델(`models/kws_tf221_calibrated.tflite`)에 기록된 float 가중치와 보정 min/max.\n")
    w("| 항목 | 결과 |\n|---|---|")
    w(f"| 활성값 scale·zero-point | {s['activation_qparams_equal']} 일치 |")
    w(f"| 가중치 scale(채널별) | {s['weight_scales_equal']} 층 일치 |")
    w(f"| int8 가중치 값 | {s['weight_count']:,}개 중 불일치 {s['weight_int8_mismatch']} |")
    w(f"| bias scale | {s['bias_scales_equal']} 층 일치 |")
    w(f"| int32 bias 값 | {s['bias_count']}개 중 불일치 {s['bias_int32_mismatch']} "
      f"(나눗셈 규칙으로 계산하면 {s['bias_int32_mismatch_divide_rule']}개 불일치) |")
    w(f"| 보정 min/max 직접 재측정 | TF 내장 런타임 {s['stats_equal_tf_runtime']}, "
      f"LiteRT {s['stats_equal_litert_runtime']} 비트 일치 |\n")
    w("MLPerf가 배포한 int8 모델과 같은 SavedModel의 변환 결과는 가중치가 거의 전부 다르다 "
      "(층마다 int8 가중치 대부분이 다르고, scale 차이가 최대 수백 %). "
      "배포 `.tflite`와 SavedModel은 서로 다른 학습 결과로 보인다 — 자세한 근거는 docs/notes.md.\n")

    w("## 3. T2·T3 — numpy 정수 커널 대 LiteRT 세 실행 경로 (비트 단위)\n")
    pol = {k: v["policy"] for k, v in be["fuzz"].items()}
    w("런타임마다 원문에서 확인한 재양자화 방식을 쓴다.\n")
    w("| 연산 | 참조 커널(BUILTIN_REF) | 최적화 커널(XNNPACK 끔) | XNNPACK |\n|---|---|---|---|")
    for kind, p in pol.items():
        w(f"| {kind} | {p['reference']} | {p['optimized']} | {p['xnnpack']} |")
    w("")
    w("| 대조 | 참조 | 최적화 | XNNPACK |\n|---|---:|---:|---:|")
    for name in ("mlperf_int8", "tf221_int8"):
        e = be[name]["end_to_end_test_set"]
        w(f"| 망 전체, {name} (테스트 {dc['test']['n']:,}개 × 출력 12) | "
          + " | ".join(f"불일치 {e[m]['mismatch']}" for m in ("reference", "optimized", "xnnpack")) + " |")
        li = be[name]["layer_isolated"]
        cells = []
        for m in ("reference", "optimized", "xnnpack"):
            bad = sum(o["mismatch"].get(o["policy"], 0) for src in ("random", "test")
                      for o in li[f"{m}/{src}"]["ops"])
            cells.append(f"불일치 {bad}")
        w(f"| 층별 13개 연산, {name} (무작위·실제 입력 각 300개) | " + " | ".join(cells) + " |")
    for kind, v in be["fuzz"].items():
        eq = v["all_outputs_equal"]
        w(f"| 무작위 {kind} 모델 {v['models']}개 (출력 전체 일치 모델 수) | "
          + " | ".join(f"{eq[m].get(pol[kind][m])}/{v['models']}" for m in ("reference", "optimized", "xnnpack")) + " |")
    w("")
    w("무작위 모델에서 다른 방식으로 계산하면 몇 개가 일치하는지(방식 구별력):\n")
    w("| 연산 · 런타임 | " + " | ".join(("double", "single", "ruy", "float", "fp32", "neon8")) + " |\n|---|"
      + "---:|" * 6)
    for kind in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
        for m in ("reference", "optimized", "xnnpack"):
            eq = be["fuzz"][kind]["all_outputs_equal"][m]
            w(f"| {kind} · {m} | " + " | ".join(str(eq.get(r, "-")) for r in
                                               ("double", "single", "ruy", "float", "fp32", "neon8")) + " |")
    sm = be["fuzz"]["SOFTMAX"]["all_outputs_equal"]
    w("| SOFTMAX · 참조/최적화/XNNPACK (참조 구현 · 최적화 구현) | "
      + " / ".join(f"{sm[m].get('reference')}·{sm[m].get('optimized')}" for m in ("reference", "optimized", "xnnpack"))
      + " | | | | | |\n")

    w("## 4. FP32 대 INT8 — 정확도·크기\n")
    w("> **통계 읽는 법.** 같은 테스트 세트에서 두 모델을 비교할 때는 McNemar 정확 검정을 쓴다 "
      "(한쪽만 맞힌 샘플 수로 판정). 이 문서의 비교는 30개 가까이 되므로, 우연을 배제하는 기준을 "
      "**p < 0.002**(본페로니 보정, 0.05/25)로 잡는다. 그보다 큰 p값의 차이는 '구별되지 않음'으로 읽는다.\n")
    ms = acc["models"]
    w("| 모델 | 테스트 정확도 | 파일 크기 | 파라미터 | MAC | x86 지연(1개, 중앙값) |\n|---|---:|---:|---:|---:|---|")
    k = ms["fp32_keras"]
    w(f"| FP32 Keras (레퍼런스 SavedModel) | {pct(k['accuracy'])} | - | {k['params_keras_total']:,} (BN 포함) | - | - |")
    for name, label in (("fp32_tflite_tf221", "FP32 TFLite (TF 2.21 변환)"),
                        ("mlperf_float32_file_dynamic_range", "MLPerf `*_float32.tflite` (실제로는 동적 범위 양자화)")):
        v = ms[name]
        lat = ", ".join(f"{m} {t:.3f} ms" for m, t in v["latency_ms"].items())
        w(f"| {label} | {pct(v['accuracy'])} | {v['bytes']:,} B | {v['params']:,} | {v['macs']:,} | {lat} |")
    for name, label in (("tf221_int8", "INT8 (TF 2.21 변환)"), ("mlperf_int8", "INT8 (MLPerf 배포)")):
        v = ms[name]
        a = v["accuracy"]["round+clamp / optimized"]
        lat = ", ".join(f"{m} {t:.3f} ms" for m, t in v["latency_ms"].items())
        w(f"| {label} | {pct(a)} | {v['bytes']:,} B | {v['params']:,} | {v['macs']:,} | {lat} |")
    w("")
    w(f"- INT8(TF 2.21) − FP32 Keras: {mc(acc['mcnemar_fp32_keras_vs_tf221_int8'], flip=True)}")
    w("- 지연시간은 x86 클라우드 VM에서 잰 값이라 마이크로컨트롤러 성능을 대표하지 않는다.\n")
    w("런타임 사이 출력 차이(같은 int8 모델·같은 입력, 테스트 4,890개 중):\n")
    w("| 모델 · 입력 변환 | 참조↔최적화: 출력이 하나라도 다른 샘플 | 참조↔XNNPACK: 출력이 다른 샘플 | 참조↔XNNPACK: top-1이 다른 샘플 |\n|---|---:|---:|---:|")
    for name in ("mlperf_int8", "tf221_int8"):
        for inp, ag in ms[name]["runtime_agreement"].items():
            w(f"| {name} · {inp} | {ag['samples_output_differs(ref vs optimized)']} | "
              f"{ag['samples_output_differs(ref vs xnnpack)']} | {ag['samples_top1_differs(ref vs xnnpack)']} |")
    w("")
    w("INT8 정확도 — 입력 변환 방식 × 런타임:\n")
    w("| 모델 | 입력 변환 | 참조 | 최적화 | XNNPACK |\n|---|---|---:|---:|---:|")
    for name in ("mlperf_int8", "tf221_int8"):
        v = ms[name]
        for inp in ("truncate(reference eval)", "round+clamp"):
            w(f"| {name} | {inp} | " + " | ".join(pct(v["accuracy"][f"{inp} / {m}"])
                                                  for m in ("reference", "optimized", "xnnpack")) + " |")
    w("")
    for name in ("mlperf_int8", "tf221_int8"):
        v = ms[name]
        w(f"- {name}: 입력값 {v['input_values_total']:,}개 중 int8 범위 밖 {v['input_values_out_of_int8_range']:,}개, "
          f"두 변환 방식이 다른 값 {v['input_values_differ_between_methods']:,}개. "
          + "; ".join(f"{k2.replace(' vs ', ' − ')}: {mc(m2)}" for k2, m2 in v["mcnemar"].items()))
    w("")

    w("## 5. 설계 변수 실험\n")
    g = exp["granularity"]
    w("### A. 가중치 채널별 대 텐서별 (실제 int8 변환)\n")
    w("| 방식 | 정확도 | 파일 크기 |\n|---|---:|---:|")
    for name in ("per_channel", "per_tensor"):
        w(f"| {name} | {pct(g[name]['accuracy'])} | {g[name]['bytes']:,} B |")
    w(f"\n채널별 − 텐서별: {mc(g['mcnemar'])}\n")
    w("채널별 모델이 13.9 KB 큰 까닭: 채널별 양자화 항목이 가중치 588 + bias 588 = 1,176개이고, "
      "flatbuffer 스키마가 scale을 float32(4 B), zero-point를 int64(8 B)로 저장해 1,176 × 12 B ≈ 14.1 KB가 "
      "더 든다(텐서별은 텐서 20개 × 12 B). zero-point는 전부 0인데도 자리를 차지한다.\n")
    w(f"INT8(채널별) − FP32 TFLite: {mc(exp['float_vs_int8'], flip=True)}\n")
    w("### B. 보정 샘플 수 (검증 분할에서 무작위 추출, 시드 "
      f"{len(exp['calibration_size']['1'])}개)\n")
    w("| 샘플 수 | 평균 | 최저 | 최고 |\n|---:|---:|---:|---:|")
    for n, v in exp["calibration_size"].items():
        if n.isdigit():
            w(f"| {n} | {pct(np.mean(v))} | {pct(min(v))} | {pct(max(v))} |")
    w(f"| 120 (MLPerf 보정 세트) | {pct(exp['calibration_size']['reference_120'])} | | |\n")
    w("![calibration size](figures/calibration_size.png)\n")
    w("### C. 가중치 대칭 대 비대칭 — 시뮬레이션\n")
    w("가중치만 양자화→역양자화하고 활성값은 float로 둔 결과다(실제 int8 추론이 아님). "
      f"기준 FP32 TFLite {pct(exp['weight_sim']['float_baseline'])}.\n")
    w("| 비트 | 단위 | 대칭 | 정확도 | 가중치 SQNR 평균(최저) | 이 설정 − FP32 |\n|---:|---|:---:|---:|---|---|")
    for r in exp["weight_sim"]["configs"]:
        w(f"| {r['bits']} | {'채널별' if r['per_channel'] else '텐서별'} | {'대칭' if r['symmetric'] else '비대칭'} | "
          f"{pct(r['accuracy'])} | {r['sqnr_db_mean']:.1f} dB ({r['sqnr_db_min']:.1f}) | {mc(r['mcnemar_vs_float'], flip=True)} |")
    w("")
    w("대칭 − 비대칭(같은 비트·단위끼리):\n")
    for r in exp["weight_sim"]["symmetric_vs_asymmetric"]:
        w(f"- {r['bits']}비트 {'채널별' if r['per_channel'] else '텐서별'}: {mc(r)}")
    w("\n![weight quantization simulation](figures/weight_quant_sim.png)\n")
    ls = exp["layer_sensitivity"]
    w(f"### D. 층별 민감도 — {ls['setting']} (시뮬레이션)\n")
    w("| 층 | 이 층만 양자화 | 이 층만 float로 남기고 나머지 양자화 |\n|---|---:|---:|")
    for r in ls["rows"]:
        w(f"| {r['op']} | {pct(r['accuracy_only_this_layer_quantized'])} | "
          f"{pct(r['accuracy_all_but_this_layer_quantized'])} |")
    rows = ls["rows"]
    worst = min(rows, key=lambda r: r["accuracy_only_this_layer_quantized"])
    cfg = next(c for c in exp["weight_sim"]["configs"]
               if c["bits"] == 4 and not c["per_channel"] and c["symmetric"])
    base = exp["weight_sim"]["float_baseline"]
    drops = [100 * (base - r["accuracy_only_this_layer_quantized"]) for r in rows if r is not worst]
    k = rows.index(worst)
    sq = cfg["sqnr_db_per_layer"]
    w(f"\n전 층을 양자화하면 {pct(cfg['accuracy'])}. 혼자서 가장 크게 떨어뜨리는 층은 {worst['op']}"
      f"({pct(worst['accuracy_only_this_layer_quantized'])})이고, 이 층만 float로 두면 "
      f"{pct(worst['accuracy_all_but_this_layer_quantized'])}까지 회복된다. 나머지 층은 혼자서는 "
      f"{min(drops):.2f}~{max(drops):.2f}%p만 떨어뜨리지만 합쳐지면 크게 떨어진다. "
      f"이 층의 가중치 SQNR은 {sq[k]:.1f} dB로, 가장 낮은 층({rows[sq.index(min(sq))]['op']}, {min(sq):.1f} dB)이 "
      "아니다 — SQNR만으로는 민감한 층을 고를 수 없다.\n")
    (RESULTS / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("→ results/results.md")


if __name__ == "__main__":
    main()
