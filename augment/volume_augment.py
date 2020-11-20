# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : volume_augment.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/18 下午3:38
import random
from utils import read_wave_from_file, save_wav, tensor_to_img, get_feature


def volume_augment(samples, min_gain_dBFS=-10, max_gain_dBFS=10):
    """
    音量增益范围约为【0.316，3.16】，不均匀，指数分布，降低幂函数的底10.可以缩小范围
    :param samples: 音频数据，一维
    :param min_gain_dBFS:
    :param max_gain_dBFS:
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    gain = random.uniform(min_gain_dBFS, max_gain_dBFS)
    gain = 10. ** (gain / 20.)
    samples = samples * gain
    # improvement:保证输出的音频还是原类型，不然耳朵会聋
    samples = samples.astype(data_type)
    return samples


if __name__ == '__main__':
    file = '../audio/speech.wav'
    audio_data, frame_rate = read_wave_from_file(file)
    feature = get_feature(audio_data)
    tensor_to_img(feature)

    audio_data = volume_augment(audio_data)

    out_file = '../audio/volume_augment.wav'
    save_wav(out_file, audio_data)
    feature = get_feature(audio_data)
    tensor_to_img(feature)
