"""Speech Commands v2 → MFCC 특징 (49×10×1). TensorFlow·tensorflow-datasets가 필요하다.

MLPerf Tiny 레퍼런스(`benchmark/training/keyword_spotting/get_dataset.py`)의 평가용 경로
(is_training=False, feature_type='mfcc')를 그대로 옮겼다. 증강(배경 잡음·시간 이동)은 평가
경로에서 꺼져 있으므로 없다. 원본과 같은 값을 내는지는 scripts/prepare_data.py가 확인한다.

레퍼런스 설정값(kws_util.py 기본값): 16 kHz, 창 30 ms(480 샘플), 보폭 20 ms(320 샘플),
멜 필터 40개(20–4000 Hz), DCT 계수 10개 → 49 프레임 × 10.
"""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

SAMPLE_RATE = 16000
DESIRED_SAMPLES = 16000
WINDOW, STRIDE = 480, 320
N_MEL, LOW_HZ, HIGH_HZ = 40, 20.0, 4000.0
N_DCT, N_FRAMES = 10, 49
LABELS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes",
          "_silence_", "_unknown_"]


def mfcc(audio):
    """int16 파형 하나 → float32 [49, 10, 1]. 레퍼런스 prepare_processing_graph와 같은 연산 순서."""
    wav = tf.cast(audio, tf.float32)
    wav = wav / tf.reduce_max(wav)                       # 레퍼런스 그대로: 절댓값이 아닌 최댓값
    wav = tf.pad(wav, [[0, DESIRED_SAMPLES - tf.shape(wav)[-1]]])
    wav = tf.multiply(wav, tf.constant(1, dtype=tf.float32))
    wav = tf.pad(wav, tf.constant([[2, 2]], tf.int32), mode="CONSTANT")   # 시간 이동 0
    wav = tf.slice(wav, tf.constant([2], tf.int32), [DESIRED_SAMPLES])
    stfts = tf.signal.stft(wav, frame_length=WINDOW, frame_step=STRIDE, fft_length=None,
                           window_fn=tf.signal.hann_window)
    spec = tf.abs(stfts)
    mel_w = tf.signal.linear_to_mel_weight_matrix(N_MEL, stfts.shape[-1], SAMPLE_RATE, LOW_HZ, HIGH_HZ)
    mel = tf.tensordot(spec, mel_w, 1)
    mel.set_shape(spec.shape[:-1].concatenate(mel_w.shape[-1:]))
    log_mel = tf.math.log(mel + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :N_DCT]
    return tf.reshape(mfccs, [N_FRAMES, N_DCT, 1])


def load_split(split, data_dir, indices=None):
    """tfds 분할 → (특징 float32 [N,49,10,1], 레이블 int64 [N]). indices를 주면 그 위치만
    (레퍼런스처럼 정렬된 순서로) 고른다."""
    ds = tfds.load("speech_commands", split=split, data_dir=str(data_dir), shuffle_files=False)
    if indices is not None:
        keep = set(int(i) for i in indices)
        audio, labels = [], []
        for i, d in enumerate(ds):
            if i in keep:
                a = d["audio"].numpy()
                audio.append(np.pad(a, (0, DESIRED_SAMPLES - len(a))))
                labels.append(int(d["label"].numpy()))
        ds = tf.data.Dataset.from_tensor_slices({"audio": audio, "label": labels})
    ds = ds.map(lambda d: (mfcc(d["audio"]), d["label"]))
    feats, labels = [], []
    for f, y in ds.batch(256):
        feats.append(f.numpy())
        labels.append(y.numpy())
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels).astype(np.int64)
