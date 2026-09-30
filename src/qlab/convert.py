"""TensorFlow 2.21 변환기로 레퍼런스 Keras 모델 → `.tflite` (float32 / full-int8).
TensorFlow와 tf-keras가 필요하다.

MLPerf Tiny 레퍼런스 quantize.py와 같은 경로(tf.keras.models.load_model → from_keras_model)와
설정(Optimize.DEFAULT, TFLITE_BUILTINS_INT8, 입·출력 int8, 보정 샘플을 1개씩 공급)을 쓴다.
TF 2.21의 기본 Keras 3는 이 옛 SavedModel을 열지 못하므로 Keras 2(tf-keras)로 전환한다
(TF_USE_LEGACY_KERAS=1 — TensorFlow를 import하기 전에 설정해야 한다).

레퍼런스와 다른 점 하나: 레퍼런스의 `*_float32.tflite`는 Optimize.DEFAULT가 켜진 채 변환돼
합성곱 가중치가 int8인 동적 범위 양자화 모델이다. 여기의 to_float는 최적화를 끈 순수 float32다.
"""
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

if os.environ["TF_USE_LEGACY_KERAS"] != "1" or not tf.keras.__name__.startswith("tf_keras"):
    raise ImportError("tf-keras(Keras 2)가 필요하다: pip install tf-keras, TF_USE_LEGACY_KERAS=1")


def load_keras(saved_model_dir):
    return tf.keras.models.load_model(str(saved_model_dir))


def to_float(model):
    """최적화 없이 순수 float32 `.tflite`."""
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def to_dynamic_range(model):
    """레퍼런스 `kws_ref_model_float32.tflite`와 같은 설정: Optimize.DEFAULT, 보정 데이터 없음."""
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    return c.convert()


def to_int8(model, calibration, *, per_channel=True, calibrate_only=False):
    """calibration: float32 [N,49,10,1]. per_channel=False는 비공개 옵션
    `_experimental_disable_per_channel`을 쓴다(버전이 바뀌면 사라질 수 있음).
    calibrate_only=True면 양자화 직전의 float 모델(텐서마다 보정 min/max가 기록됨)을 돌려준다."""
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    data = np.asarray(calibration, np.float32)
    c.representative_dataset = lambda: ([x[None]] for x in data)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    if not per_channel:
        c._experimental_disable_per_channel = True
    if calibrate_only:
        c._experimental_calibrate_only = True
    return c.convert()
