# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : frequency_mask.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/20 下午3:25
import random
import numpy as np
from utils import read_wave_from_file, get_feature, tensor_to_img


def frequency_mask_augment(inputs, max_mask_frequency=5, mask_num=5):
    """

    :param inputs: 三维numpy或tensor，(batch, time_step,  feature_dim)
    :param max_mask_frequency:
    :param mask_num:
    :return:
    """
    dim = len(inputs.shape)
    feature_len = 0
    if dim == 2:
        feature_len = inputs.shape[1]
    elif dim == 3:
        feature_len = inputs.shape[2]
    for i in range(mask_num):
        f = np.random.uniform(low=0.0, high=max_mask_frequency)
        f = int(f)
        f0 = random.randint(0, feature_len - f)
        if dim == 2:
            inputs[:, f0:f0 + f] = 0
        elif dim == 3:
            inputs[:, :, f0:f0 + f] = 0
    return inputs


if __name__ == '__main__':
    audio_path = '../audio/speech.wav'
    audio, sampling_rate = read_wave_from_file(audio_path)
    feature = get_feature(audio, sampling_rate, 128)
    feature_1 = feature.copy()

    tensor_to_img(feature)
    print(feature.shape)
    feature = frequency_mask_augment(feature, max_mask_frequency=5, mask_num=10)
    tensor_to_img(feature)
