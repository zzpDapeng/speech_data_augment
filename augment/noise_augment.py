# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : noise_augment.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/20 上午10:07
import numpy as np
from utils import read_wave_from_file, save_wav, tensor_to_img, get_feature


def nature_noise_augmentation(samples, noise_list, max_db=0.5):
    """
    叠加自然噪声
    :param samples: 语音采样
    :param noise_list:噪声文件列表
    :param max_db:最大噪声增益
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    noise_path = np.random.choice(noise_list)
    # 随机音量
    db = np.random.uniform(low=0.1, high=max_db)
    aug_noise, fs = read_wave_from_file(noise_path)
    # 噪声片段增长
    while len(aug_noise) <= len(samples):
        aug_noise = np.concatenate((aug_noise, aug_noise), axis=0)
    # 随机位置开始截取与语音数据等长的噪声数据
    diff_len = len(aug_noise) - len(samples)
    start = np.random.randint(0, diff_len)
    end = start + len(samples)
    # 叠加
    samples = samples + db * aug_noise[start:end]
    samples = samples.astype(data_type)
    return samples


def uniform_white_noise_numpy(samples, min_db=10, max_db=200):
    """
    均匀白噪声
    :param samples:
    :param max_db:
    :param min_db:
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    db = np.random.randint(low=min_db, high=max_db)
    noise = np.random.uniform(low=-db, high=db, size=len(samples))  # 高斯分布
    samples = samples + noise
    samples = samples.astype(data_type)
    return samples


def gaussian_white_noise_numpy(samples, min_db=10, max_db=200):
    """
    高斯白噪声
    噪声音量db
        db = 10, 听不见
        db = 100,可以听见，很小
        db = 500,大
        人声都很清晰
    :param samples:
    :param max_db:
    :param min_db:
    :return:
    """
    samples = samples.copy()  # frombuffer()导致数据不可更改因此使用拷贝
    data_type = samples[0].dtype
    db = np.random.randint(low=min_db, high=max_db)
    noise = db * np.random.normal(0, 1, len(samples))  # 高斯分布
    samples = samples + noise
    samples = samples.astype(data_type)
    return samples


if __name__ == '__main__':
    file = '../audio/speech.wav'
    audio_data, _ = read_wave_from_file(file)
    feature = get_feature(audio_data)
    tensor_to_img(feature)

    audio_data = gaussian_white_noise_numpy(audio_data)

    out_file = '../audio/uniform_white_noise_numpy.wav'
    feature = get_feature(audio_data)
    save_wav(out_file, audio_data)
    tensor_to_img(feature)
