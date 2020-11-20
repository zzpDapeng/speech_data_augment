# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : pitch_augment.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/18 下午9:44
import librosa
import random
import cv2
from utils import read_wave_from_file, save_wav, tensor_to_img, get_feature


def pitch_librosa(samples, sr=16000, ratio=5):
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    samples = samples.astype('float')
    ratio = random.uniform(-ratio, ratio)
    samples = librosa.effects.pitch_shift(samples, sr, n_steps=ratio)
    samples = samples.astype(data_type)
    return samples


def pitch_cv(samples, min_pitch=0.5, max_pitch=1.5):
    """
    不好用,会丢失大量语音信息,而且范围不好调整
    :param samples:
    :param min_pitch:
    :param max_pitch:
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    length = samples.shape[0]
    data_type = samples[0].dtype
    ratio = random.uniform(min_pitch, max_pitch)
    samples = cv2.resize(samples, (1, int(length / ratio))).squeeze()
    length_change = len(samples) - length
    samples = samples[int(length_change / 2):int(length_change / 2) + length]
    samples = samples.astype(data_type)
    return samples


if __name__ == '__main__':
    file = '../audio/speech.wav'
    audio_data, frame_rate = read_wave_from_file(file)
    feature = get_feature(audio_data)
    tensor_to_img(feature)

    audio_data = pitch_cv(audio_data)

    out_file = '../audio/pitch_librosa.wav'
    save_wav(out_file, audio_data)
    feature = get_feature(audio_data)
    tensor_to_img(feature)
