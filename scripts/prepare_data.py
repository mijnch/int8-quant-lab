"""Speech Commands v2를 받아 MFCC 특징을 data/features.npz로 저장한다.

    python scripts/prepare_data.py --tfds-dir <tfds 저장 폴더> [--mlperf-dir <mlcommons/tiny 클론>]

--mlperf-dir를 주면 MLPerf Tiny 레퍼런스 코드(get_dataset.get_training_data)로도 같은 분할의
특징을 만들어, 이 저장소의 구현(src/qlab/data.py)과 값이 같은지 대조해 results/data_check.json에 남긴다.

주의: tensorflow-datasets의 기본 주소(download.tensorflow.org)가 막힌 네트워크에서는
--gcs-mirror를 주면 같은 파일을 storage.googleapis.com에서 받는다(sha256은 tfds가 등록한 값과
따로 대조할 것 — README 참고).
"""
import argparse
import os
import sys

from _common import DATA, MODELS, save_json

import numpy as np


def reference_features(mlperf_dir, tfds_dir):
    """MLPerf 레퍼런스 코드로 test·validation·보정 분할의 특징을 만든다."""
    kws = os.path.join(mlperf_dir, "benchmark", "training", "keyword_spotting")
    sys.path.insert(0, kws)
    cwd = os.getcwd()
    os.chdir(kws)                      # quant_cal_idxs.txt를 상대 경로로 읽는다
    try:
        import get_dataset
        import kws_util
        sys.argv = ["x", "--data_dir", str(tfds_dir), "--batch_size", "256"]
        flags, _ = kws_util.parse_command()
        _, ds_test, ds_val = get_dataset.get_training_data(flags)
        _, _, ds_cal = get_dataset.get_training_data(flags, val_cal_subset=True)
    finally:
        os.chdir(cwd)

    def collect(ds):
        xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in ds])
        return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)
    return {"test": collect(ds_test), "val": collect(ds_val), "cal": collect(ds_cal)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfds-dir", required=True)
    ap.add_argument("--mlperf-dir")
    ap.add_argument("--gcs-mirror", action="store_true")
    a = ap.parse_args()

    if a.gcs_mirror:
        from tensorflow_datasets.datasets.speech_commands import speech_commands_dataset_builder as b
        g = "https://storage.googleapis.com/download.tensorflow.org/data/"
        b._DOWNLOAD_PATH = g + "speech_commands_v0.02.tar.gz"
        b._TEST_DOWNLOAD_PATH_ = g + "speech_commands_test_set_v0.02.tar.gz"
    import tensorflow_datasets as tfds
    tfds.builder("speech_commands", data_dir=a.tfds_dir).download_and_prepare()

    from qlab import data
    cal_idx = np.loadtxt(MODELS / "quant_cal_idxs.txt", dtype=np.int64)
    mine = {"test": data.load_split("test", a.tfds_dir),
            "val": data.load_split("validation", a.tfds_dir),
            "cal": data.load_split("validation", a.tfds_dir, indices=np.sort(cal_idx))}
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / "features.npz", **{f"{k}_x": v[0] for k, v in mine.items()},
             **{f"{k}_y": v[1] for k, v in mine.items()})
    print({k: v[0].shape for k, v in mine.items()})

    if a.mlperf_dir:
        ref = reference_features(a.mlperf_dir, a.tfds_dir)
        report = {}
        for k in mine:
            (mx, my), (rx, ry) = mine[k], ref[k]
            report[k] = {"n": int(len(my)), "labels_equal": bool(np.array_equal(my, ry)),
                         "features_bit_equal": bool(mx.shape == rx.shape and
                                                    np.array_equal(mx.view(np.uint32), rx.view(np.uint32))),
                         "max_abs_diff": float(np.abs(mx - rx).max()) if mx.shape == rx.shape else None}
        report["label_counts_test"] = np.bincount(mine["test"][1], minlength=12).tolist()
        save_json("data_check.json", report)
        print(report)


if __name__ == "__main__":
    main()
