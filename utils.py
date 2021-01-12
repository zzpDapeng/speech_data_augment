# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : utils.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/20 下午3:19
import wave
import librosa
import numpy as np
import matplotlib.pyplot as plt


def tensor_to_img(spectrogram, x_range=None, y_range=None):
    plt.figure()  # arbitrary, looks good on my screen.
    # plt.imshow(spectrogram[0].T)
    plt.imshow(spectrogram.T)
    if x_range is not None:
        plt.xlim(0, x_range)
    if y_range is not None:
        plt.ylim(0, y_range)
    plt.show()


# 绘制频谱图
def plot_spectrogram(spec, note):
    """
    audio feature figure
    (feature_dim, time_step)
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()


def save_wav(file_name, audio_data, channels=1, sample_width=2, rate=16000):
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(b''.join(audio_data))
    wf.close()


def read_wave_from_file(audio_file):
    """
    return 一维numpy数组，如（584,） 采样率"""
    wav = wave.open(audio_file, 'rb')
    num_frames = wav.getnframes()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)
    return wave_data, framerate


def concat_frame(features, left_context_width, right_context_width):
    time_steps, features_dim = features.shape
    concated_features = np.zeros(
        shape=[time_steps, features_dim *
               (1 + left_context_width + right_context_width)],
        dtype=np.float32)
    # middle part is just the uttarnce
    concated_features[:, left_context_width * features_dim:
                         (left_context_width + 1) * features_dim] = features

    for i in range(left_context_width):
        # add left context
        concated_features[i + 1:time_steps,
        (left_context_width - i - 1) * features_dim:
        (left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

    for i in range(right_context_width):
        # add right context
        concated_features[0:time_steps - i - 1,
        (right_context_width + i + 1) * features_dim:
        (right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

    return concated_features


def subsampling(features, subsample=3):
    interval = subsample
    temp_mat = [features[i]
                for i in range(0, features.shape[0], interval)]
    subsampled_features = np.row_stack(temp_mat)
    return subsampled_features


def get_feature(wave_data, framerate=16000, feature_dim=128):
    """
    :param wave_data: 一维numpy,dtype=int16
    :param framerate:
    :param feature_dim:
    :return: specgram [序列长度,特征维度]
    """
    wave_data = wave_data.astype("float32")
    specgram = librosa.feature.melspectrogram(wave_data, sr=framerate, n_fft=512, hop_length=160, n_mels=feature_dim)
    specgram = np.where(specgram == 0, np.finfo(float).eps, specgram)
    specgram = np.log10(specgram)
    return specgram


def get_final_feature(samples, sample_rate=16000, feature_dim=128, left=3, right=0, subsample=3):
    feature = get_feature(samples, sample_rate, feature_dim)
    feature = concat_frame(feature, left, right)
    feature = subsampling(feature, subsample)
    return feature


def log_mel(file, sr=16000, dim=80, win_len=25, stride=10):
    samples, sr = librosa.load(file, sr=sr)
    samples = samples * 32768
    win_len = int(sr / 1000 * win_len)
    hop_len = int(sr / 1000 * stride)
    feature = librosa.feature.melspectrogram(samples, sr=sr, win_length=win_len, hop_length=hop_len, n_mels=dim)
    feature = np.where(feature == 0, np.finfo(float).eps, feature)
    feature = np.log10(feature)
    return feature
