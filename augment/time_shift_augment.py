# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : time_shift_augment.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/18 下午9:58
import numpy as np
from utils import read_wave_from_file, save_wav, tensor_to_img, get_feature


def time_shift(samples, max_ratio=0.05):
    """
    改进:
    1.为在一定比例范围内随机偏移，不再需要时间
    2.循环移动
    :param samples: 音频数据
    :param max_ratio:
    :return:
    """
    samples = samples.copy()
    frame_num = samples.shape[0]
    max_shifts = frame_num * max_ratio  # around 5% shift
    # shifts_num = np.random.randint(-max_shifts, max_shifts)
    shifts_num = int(max_shifts)
    print(shifts_num)
    if shifts_num > 0:
        # time advance
        temp = samples[:shifts_num]
        samples[:-shifts_num] = samples[shifts_num:]
        # samples[-shifts_num:] = 0
        samples[-shifts_num:] = temp
    elif shifts_num < 0:
        # time delay
        temp = samples[shifts_num:]
        samples[-shifts_num:] = samples[:shifts_num]
        # samples[:-shifts_num] = 0
        samples[:-shifts_num] = temp
    return samples


def time_shift_numpy(samples, max_ratio=0.05):
    """
    时间变化是在时间轴的±5％范围内的随机滚动。环绕式转换以保留所有信息。
    Shift a spectrogram along the frequency axis in the spectral-domain at random
    :param max_ratio:
    :param samples: 音频数据，一维(序列长度,) 或 特征数据(序列长度,特征维度)
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    frame_num = samples.shape[0]
    max_shifts = frame_num * max_ratio  # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)
    samples = np.roll(samples, nb_shifts, axis=0)
    samples = samples.astype(data_type)
    return samples


if __name__ == '__main__':
    file = '../audio/speech.wav'
    audio_data, frame_rate = read_wave_from_file(file)
    feature = get_feature(audio_data)
    tensor_to_img(feature)

    # audio_data = time_shift(audio_data)
    audio_data = time_shift_numpy(audio_data)

    out_file = '../audio/time_shift_numpy.wav'
    feature = get_feature(audio_data)
    save_wav(out_file, audio_data)
    tensor_to_img(feature)
