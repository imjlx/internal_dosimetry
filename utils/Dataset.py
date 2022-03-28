#!/usr/bin/env python
# encoding: utf-8
"""
    @File       : Dataset.py
    @Time       : 2022/3/8 17:31
    @Author     : Haoran Jia
    @license    : Copyright(c) 2022 Haoran Jia. All rights reserved.
    @contact    : 21211140001@m.fudan.edu.cn
    @Description：
"""
import os.path

import tensorflow as tf
import numpy as np

from typing import List, Dict, Tuple


class PatientDatasetProcessor(object):

    # project_path = r"E:\JHR\internal_dosimetry"
    project_path = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, ID):
        self.ID: int = ID
        self.patient_folder: str = self.project_path + "\\dataset\\patient" + str(ID)
        self.patch_folder: str = self.project_path + "\\dataset\\patient" + str(ID) + "\\patch_norm"
        self.n_patches: int = 0
        self.ds = None
        self.coefficient = None

    def create_dataset(self, particle="positron", energy=0.2498):
        """
        创建一名患者的dataset
        :param particle: 放射源粒子种类
        :param energy: 能量
        :return: ds, (ct, pet, source, dm)
        """
        # 读取文件名
        ct = tf.data.Dataset.list_files(os.path.join(self.patch_folder, "ct/*.npy"), shuffle=False)
        pet = tf.data.Dataset.list_files(os.path.join(self.patch_folder, "pet/*.npy"), shuffle=False)
        dm = tf.data.Dataset.list_files(os.path.join(self.patch_folder, "dosemap_F18/*.npy"), shuffle=False)
        # ct = list(tf.data.Dataset.as_numpy_iterator(ct))

        # 使用load函数加载数据集
        ct = ct.map(lambda x: tf.py_function(func=self._load, inp=[x], Tout=tf.float32))
        pet = pet.map(lambda x: tf.py_function(func=self._load, inp=[x], Tout=tf.float32))
        dm = dm.map(lambda x: tf.py_function(func=self._load, inp=[x], Tout=tf.float32))
        # ct = list(tf.data.Dataset.as_numpy_iterator(ct))

        # 生成放射源相关的输入
        self.n_patches = len(os.listdir(os.path.join(self.patch_folder, "ct")))
        source = tf.data.Dataset.from_tensors(self._radio_source_tensor(particle, energy)).repeat(self.n_patches)
        # source = list(tf.data.Dataset.as_numpy_iterator(source))

        # 组合为完整的患者的数据集
        # self.ds = tf.data.Dataset.zip((ct, pet, source, dm))
        self.ds = tf.data.Dataset.zip(((ct, pet, source), dm))
        # self.ds = list(tf.data.Dataset.as_numpy_iterator(self.ds))
        return self.ds

    @staticmethod
    def _load(fpath):
        """
        tensorflow创建数据集的时候，调用以从文件名读取为数组, 并初始化
        :param fpath: npy文件地址(实际传入的是tf中的特殊结构, 要使用.numpy()提取作为内容的字符串)
        :return: 读取并处理后的的numpy数组
        """
        img = np.load(fpath.numpy())    # 根据fpath读取npy文件
        img = np.expand_dims(img, 3)    # 增加一个维度, 变为(xx, xx, xx, 1)
        return img

    @staticmethod
    def _radio_source_tensor(particle, energy):
        if particle == "electron":
            n = 0
        elif particle == "positron":
            n = 1
        elif particle == "proton":
            n = 2
        elif particle == "neutron":
            n = 3
        elif particle == "alpha":
            n = 4
        else:
            raise KeyError("unsupported particle")
        source = np.zeros(shape=[8, 8, 8, 5])
        source[:, :, :, n] = energy
        return tf.constant(source, dtype=tf.float32)


def create_dataset(patient_IDs: tuple, batch_size: int):

    # 第一个患者的
    processor = PatientDatasetProcessor(patient_IDs[0])
    ds = processor.create_dataset()
    n_patch = processor.n_patches

    # 添加后续的患者
    for ID in patient_IDs[1:]:
        processor = PatientDatasetProcessor(ID)
        ds = ds.concatenate(processor.create_dataset())
        n_patch += processor.n_patches

    # 打乱, 取batch
    ds = ds.shuffle(buffer_size=64).batch(batch_size)
    # ds_list = list(tf.data.Dataset.as_numpy_iterator(ds))
    return ds


if __name__ == '__main__':
    # p = PatientDatasetProcessor(1)
    # p.create_dataset()
    ds = create_dataset((1,), batch_size=8)
    print(ds)


