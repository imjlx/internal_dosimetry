#!/usr/bin/env python
# encoding: utf-8
"""
    @File       : Data.py
    @Author     : Haoran Jia
    @license    : Copyright(c) 2022 Haoran Jia. All rights reserved.
    @contact    : 21211140001@fudan.m.edu.cn
    @Description：
"""

import re
import numpy as np
import os
import numba as nb
import tqdm
import time
import SimpleITK as sitk
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

from utils import Visual
from utils.RemoveBedFilter import RemoveBedFilter

"""
读取原始文件的方法
"""


@nb.jit(parallel=True)
def create_ct_without_bed(ct_origin, atlas, ct_AutoRemoveBed):
    """
    利用分割数据(不确定是否完全包含人体) 和 自动去床(丢失肺部)结果, 生成有肺部的, 去掉床的CT图像
    :param ct_origin: 原始CT
    :param atlas: 分割数据
    :param ct_AutoRemoveBed: 自动去床CT
    :return: 生成的结果CT
    """
    # 对所有体素遍历
    for i in range(atlas.shape[0]):
        for j in range(atlas.shape[1]):
            for k in range(atlas.shape[2]):
                # 如果是atlas中的组织, 就用原始ct的值覆盖自动去床ct的值, 找回肺部
                if atlas[i, j, k, 0] > 0:
                    ct_AutoRemoveBed[i, j, k, 0] = ct_origin[i, j, k, 0]

    return ct_AutoRemoveBed


@nb.jit(nopython=True, parallel=True)
def _create_dosemap_pet_without_air_SimpleThresholdOnCT(img: np.ndarray, ct: np.ndarray, threshold: int) -> np.ndarray:
    """
    设定一个最低的阈值, 低于该阈值的CT均认为是背景, 并将对应的dosemap设置为0
    :param img: 处理的dosemap
    :param ct: 参考的ct
    :param threshold: 阈值
    :return: 处理后的dosemap
    """
    for i, pixel in enumerate(ct.flat):
        if pixel < threshold:
            img.flat[i] = 0
    return img


@nb.jit(parallel=True)
def _create_patch_index_array(img: np.ndarray, size: int, step: int, ratio: float) -> np.ndarray:
    """
    根据img生成可取的块的坐标, 以0为背景, 大于0的值为前景
    :param img: 参考轮廓图
    :param size: 块的大小
    :param step: 块的间距
    :param ratio: 是否取的依据: 组织的占比
    :return: 可取块的顶点坐标
    """
    # 计算每个维度可取的个数
    n_i = int(np.floor((img.shape[0] - size) / step) + 1)
    n_j = int(np.floor((img.shape[1] - size) / step) + 1)
    n_k = int(np.floor((img.shape[2] - size) / step) + 1)

    index_list = []

    for i in range(n_i):
        i *= step
        for j in range(n_j):
            j *= step
            for k in range(n_k):
                k *= step
                # i,j,k为块顶点的坐标, 取出此块
                img_patch = img[i:i + size, j:j + size, k:k + size]
                if np.sum(img_patch > 0) / img_patch.size > ratio:
                    index_list.append([i, j, k])
    return np.array(index_list)


class PatientDataProcessor(object):
    """
    用于对一位患者的图像数据进行处理
    """
    project_path = r"E:\internal_dosimetry"

    def __init__(self, ID: int):
        # 患者ID和患者文件夹路径
        self.ID = ID
        self.patient_folder = PatientDataProcessor.project_path + r"\dataset\patient" + str(ID)

        # 生成的patch的个数
        self.n_patch = None

    def Execute(self):
        self.create_npy_origin(recreate=True)
        print("移除CT床")
        self.create_ct_without_bed()
        print("移除PET, dosemap空气")
        self.create_dosemap_pet_without_air(pet="pet_origin", dm="dm_origin")
        print("切块")
        self.create_patches(ct="ct_AirRemoved", pet="pet_AirRemoved", dm="dm_AirRemoved")
        print("Patient finished")

    # 测试中...
    @staticmethod
    def resample(self, readPath, savePath):
        reader = sitk.ImageFileReader()
        reader.SetFileName(readPath)
        img = reader.Execute()
        print(img.GetSize())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing((1.52344, 1.52344, 1.52344))
        new_size = [img.GetSize()[i] * img.GetSpacing()[i] / 1.5234 for i in range(3)]
        print(new_size)
        resampler.SetSize(tuple(new_size))

        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(sitk.TranslationTransform(3))
        resampler.SetOutputPixelType(sitk.sitkInt16)
        # resampler.SetDefaultPixelValue(-1024)
        img = resampler.Execute(img)
        print(img.GetSize())

        # sitk.WriteImage(img, savePath)

    # 基本文件的读取与保存
    def create_npy_origin(self, recreate: bool = False, **kwargs) -> None:
        """
        读取原始的hdr文件, 生成原始的npy文件,
        (一般)形状: (512, 512, xxx), 不设第四个维度
        数据类型: uint8, int16
        :param recreate: 是否重新生成并覆盖
        :param kwargs: 可选['ct', 'pet', 'atlas', 'dm'], 设置读取什么文件, 默认都生成
        :return: 无, 保存文件为xxx_origin.npy
        """
        # 创建npy文件夹
        dirname = self.patient_folder + "/npy"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        # 处理不完全输入kwarg的情况: 默认全部生成
        for key in ['ct', 'pet', 'atlas', 'dm']:
            if key not in kwargs.keys():
                kwargs[key] = True
        # 路径设置
        rpath = {'ct': os.path.join(self.patient_folder, "hdr/ct.hdr"),
                 'pet': os.path.join(self.patient_folder, "hdr/pet.hdr"),
                 'atlas': os.path.join(self.patient_folder, "hdr/atlas.hdr"),
                 'dm': os.path.join(self.patient_folder, "dosemap_F18/dosemap.hdr")}
        spath = {'ct': os.path.join(self.patient_folder, "npy/ct_origin.npy"),
                 'pet': os.path.join(self.patient_folder, "npy/pet_origin.npy"),
                 'atlas': os.path.join(self.patient_folder, "npy/atlas_origin.npy"),
                 'dm': os.path.join(self.patient_folder, "dosemap_F18/dm_origin.npy")}
        # 读取, 提取出ndarray, 简单处理, 保存文件
        for key in ['ct', 'pet', 'atlas', 'dm']:
            if kwargs[key] and ((not os.path.isfile(spath[key])) or recreate):  # 要处理该文件, 且不存在或要刷新
                img = sitk.ReadImage(fileName=rpath[key])  # 读取的原始文件均为16-bit signed integer
                # print(img.GetPixelIDTypeAsString())
                if key == 'atlas':
                    img = sitk.GetArrayFromImage(img).T.astype(np.uint8)  # atlas保存为无符号8位整型(np.uint8)
                else:
                    img = sitk.GetArrayFromImage(img).T  # ct, pet, dosemap均保存为16位整型(np.int16)
                np.save(spath[key], img)
                # print(img.dtype)
                print(f"Patient{self.ID}'s {key} npy origin file created.")

    def _load_npy(self, fname: str) -> np.ndarray:
        """
        根据fname读取npy文件并返回
        :param fname: 文件名, 不需要路径, 不需要后缀, 必须是符合保存规则的文件
        :return: 读取的npy数组
        """
        if re.match(pattern="^dm", string=fname):
            return np.load(self.patient_folder + "\\dosemap_F18\\" + fname + ".npy")
        else:
            return np.load(self.patient_folder + "\\npy\\" + fname + ".npy")

    # 文件进阶处理
    def create_ct_without_bed(self):
        remove_bed_filter = RemoveBedFilter(self.patient_folder + "\\hdr" + "\\ct.hdr")
        print("Generating ct without bed...")
        ct_AirRemoved = remove_bed_filter.Execute()
        # sitk.Show(ct_AirRemoved)
        ct_AirRemoved = sitk.GetArrayFromImage(ct_AirRemoved).T     # 转换为ndarray
        np.save(os.path.join(self.patient_folder, "npy/ct_AirRemoved.npy"), ct_AirRemoved)
        print(f"Patient{self.ID}'s ct_AirRemoved.npy file created.")

    def create_dosemap_pet_without_air(self, dm: str = None, pet: str = None,
                                       reference_image: str = "ct_AirRemoved",
                                       method: str = "MaskedCT", **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        去除dosemap中非人体部分的噪声值
        :param dm: 原始dosemap
        :param pet: 原始dosemap
        :param reference_image: 参考图像, 方法不同参考图像含义不同
        :param method: 选用的不同方法
        :param kwargs: 方法的特定参数
        :return: 处理后的dosemap, 并保存
        """
        if dm is not None:
            dm = self._load_npy(dm)
        if pet is not None:
            pet = self._load_npy(pet)
        # 不同的处理方法
        if method == "SimpleThresholdOnCT":
            """
            设定一个最低的阈值, 低于该阈值的CT均认为是背景, 并将对应的dosemap设置为0
            """
            ct = self._load_npy(reference_image)    # 此时reference_img是ct
            if 'threshold' not in kwargs.keys():    # 如果没有指定threshold, 则设置为900
                kwargs['threshold'] = 900
            # 利用类外numba加速的函数处理
            try:
                dm = _create_dosemap_pet_without_air_SimpleThresholdOnCT(img=dm, ct=ct, threshold=kwargs['threshold'])
            except TypeError:
                print("Haven't assign dosemap to process.")
            try:
                pet = _create_dosemap_pet_without_air_SimpleThresholdOnCT(img=pet, ct=ct, threshold=kwargs['threshold'])
            except TypeError:
                print("Haven't assign pet to process.")
        elif method == "MaskedCT":
            """
            利用去除背景信息和床的CT, CT背景值为-1024
            """
            mask = self._load_npy(reference_image)
            try:
                dm[mask == -1024] = 0
            except TypeError:
                print("Haven't assign dosemap to process.")
            try:
                pet[mask == -1024] = 0
            except TypeError:
                print("Haven't assign pet to process.")
        else:
            raise TypeError("unsupported method")

        # sitk.Show(sitk.GetImageFromArray(dm))
        if dm is not None:
            np.save(file=os.path.join(self.patient_folder, "dosemap_F18\\dm_AirRemoved.npy"), arr=dm)
            print(f"Patient{self.ID}'s dm_AirRemoved.npy file created.")
        if pet is not None:
            np.save(file=os.path.join(self.patient_folder, "npy\\pet_AirRemoved.npy"), arr=pet)
            print(f"Patient{self.ID}'s pet_AirRemoved.npy file created.")

        return pet, dm

    # 生成patch
    def create_patch_index_array(self, reference_img: np.ndarray, size: int, step: int, ratio: float) -> np.ndarray:
        """
        根据reference_img生成可取的块的坐标, 并保存
        :param reference_img: 参考轮廓遮罩图
        :param size: 块的大小
        :param step: 块的间距
        :param ratio: 是否取的依据: 组织的占比
        :return: 可取块的顶点坐标
        """
        # 创建patch文件夹
        if not os.path.exists(self.patient_folder + "\\patch"):
            os.makedirs(self.patient_folder + "\\patch")
            print("新创建patch文件夹")
        # 创建index_array.npy
        print("正在生成分割方法 index_array....")
        time_start = time.time()
        index_array = _create_patch_index_array(img=reference_img, size=size, step=step, ratio=ratio)
        np.save(file=self.patient_folder + "\\patch\\index_array.npy", arr=index_array)
        time_end = time.time()
        print(f"已生成并保存, 用时{time_end - time_start}s")

        self.n_patch = len(index_array)
        return index_array

    def create_patches(self, ct: str, pet: str, dm: str, reference_img: np.ndarray = None,
                       size: int = 128, step: int = 16, ratio: float = 0.5,
                       recreate_index_array: bool = False) -> None:
        """
        分割图像
        :param ct: 待分割的ct
        :param pet: 待分割的pet
        :param dm: 待分割的dosemap
        :param reference_img: 用于生成分割方法的参考图像, 默认为atlas
        :param size: 块的大小
        :param step: 块的间距
        :param ratio: 是否取的依据: 组织的占比
        :param recreate_index_array: 是否重新生成分割方案
        :return: 无
        """

        # 创建保存patch的文件夹
        folder_path_ct = self.patient_folder + "/patch/ct"
        folder_path_pet = self.patient_folder + "/patch/pet"
        folder_path_dm = self.patient_folder + "/patch/dosemap_F18"
        for dirname in [folder_path_ct, folder_path_pet, folder_path_dm]:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        # 设置默认参考轮廓--去背景后的ct
        if reference_img is None:
            reference_img = self._load_npy("ct_AirRemoved") + 1024  # +1024使背景值为0
        # 生成/读取index_array
        fpath_index_array = self.patient_folder + "\\patch\\index_array.npy"
        if (not os.path.isfile(fpath_index_array)) or recreate_index_array:
            index_array = self.create_patch_index_array(reference_img, size, step, ratio)
        else:
            index_array = np.load(fpath_index_array)
            print("已加载分割方法")

        # 加载数据
        ct = self._load_npy(ct)
        pet = self._load_npy(pet)
        dm = self._load_npy(dm)
        # 提取并保存为npy文件
        n = 0
        with tqdm.tqdm(index_array) as bar:
            for i, j, k in bar:
                bar.set_description("Saving .npy file")
                n += 1
                np.save(os.path.join(folder_path_ct, str(n) + ".npy"), ct[i:i + size, j:j + size, k:k + size])
                np.save(os.path.join(folder_path_pet, str(n) + ".npy"), pet[i:i + size, j:j + size, k:k + size])
                np.save(os.path.join(folder_path_dm, str(n) + ".npy"), dm[i:i + size, j:j + size, k:k + size])

    # 数据分析
    def hist(self):
        plt.style.use("seaborn-paper")
        gs = dict(top=0.9, bottom=0.05, hspace=0.25, left=0.1, right=0.95)
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), gridspec_kw=gs)
        # ct
        img = self._load_npy("ct_AirRemoved")
        Visual.hist(axes[0], img, range=[-800, 1000], bins=200, title="CT", min_base=-1024)
        # pet
        img = self._load_npy("pet_AirRemoved")
        Visual.hist(axes[1], img, bins=200, title="PET", min_base=0)
        # dosemap
        img = self._load_npy("dm_AirRemoved")
        Visual.hist(axes[2], img, bins=200, title="Dosemap", min_base=0)

        fig.suptitle(f"Histogram: Patient {self.ID}", fontsize='xx-large')
        fig.show()
        fig.savefig(fname=os.path.join(self.patient_folder, "histogram.png"), dpi=300)
        print("figure saved")
        pass

    def check_patch(self, n: int):
        ct = np.load(self.patient_folder + "\\patch\\ct\\" + str(n) + ".npy")
        pet = np.load(self.patient_folder + "\\patch\\pet\\" + str(n) + ".npy")
        dm = np.load(self.patient_folder + "\\patch\\dosemap_F18\\" + str(n) + ".npy")

        sitk.Show(sitk.GetImageFromArray(ct), "ct")
        sitk.Show(sitk.GetImageFromArray(pet), "pet")
        sitk.Show(sitk.GetImageFromArray(dm), "dosemap")
        pass



if __name__ == '__main__':
    # p = PatientDataProcessor(1)
    # p.create_npy_origin(recreate=True)
    # p.create_ct_without_bed()
    # p.create_dosemap_pet_without_air(pet="pet_origin", dm="dm_origin", reference_image="ct_AirRemoved", method='MaskedCT')
    # p.create_patches(ct="ct_AirRemoved", pet="pet_AirRemoved", dm="dm_AirRemoved", recreate_index_array=False)
    # p.check_patch(2)

    p = PatientDataProcessor(2)
    p.hist()


