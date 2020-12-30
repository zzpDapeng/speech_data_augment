# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : time_mask.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/20 下午3:25
import random
import numpy as np
from utils import read_wave_from_file, get_feature, tensor_to_img


def time_mask_augment(inputs, max_mask_time=5, mask_num=10):
    """
    时间遮掩，
    :param inputs: 三维numpy或tensor，(batch, time_step,  feature_dim)
    :param max_mask_time:
    :param mask_num:
    :return:
    """
    dim = len(inputs.shape)
    time_len = 0
    if dim == 2:
        time_len = inputs.shape[0]
    elif dim == 3:
        time_len = inputs.shape[1]
    for i in range(mask_num):
        t = np.random.uniform(low=0.0, high=max_mask_time)
        t = int(t)
        t0 = random.randint(0, time_len - t)
        if dim == 2:
            inputs[t0:t0 + t, :] = inputs[0, 0].copy()
        elif dim == 3:
            inputs[:, t0:t0 + t, :] = inputs[0, 0, 0].copy()

    return inputs


if __name__ == '__main__':
    audio_path = '../audio/speech.wav'
    audio, sampling_rate = read_wave_from_file(audio_path)
    feature = get_feature(audio, sampling_rate, 128)
    feature = feature[None, :, :]
    feature_1 = feature.copy()

    tensor_to_img(feature)
    feature = time_mask_augment(feature, max_mask_time=5, mask_num=10)
    tensor_to_img(feature)
