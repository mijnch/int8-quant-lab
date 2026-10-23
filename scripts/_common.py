"""스크립트 공통: 저장소 경로, src를 import 경로에 넣기, 결과 JSON 읽고 쓰기."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DATA = ROOT / "data"
REF_INT8 = MODELS / "kws_ref_model.tflite"
REF_FP32 = MODELS / "kws_ref_model_float32.tflite"


def environment():
    """측정 결과와 함께 남길 실행 환경(라이브러리 판). 보고서는 이 기록을 그대로 옮긴다."""
    import importlib.metadata as md
    import platform

    def ver(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{n: ver(n) for n in ("tensorflow", "tf-keras", "ai-edge-litert", "numpy")}}


def save_json(name, obj):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"→ {path.relative_to(ROOT)}")


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_features():
    import numpy as np
    path = DATA / "features.npz"
    if not path.exists():
        sys.exit("data/features.npz가 없습니다. 먼저 scripts/prepare_data.py를 실행하세요.")
    return dict(np.load(path))
