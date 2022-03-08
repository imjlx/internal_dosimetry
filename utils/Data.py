import re

import numpy as np
import os
import nibabel as nib
# import tensorflow as tf
import numba as nb
import pandas as pd
import tqdm
import time
import SimpleITK as sitk
from typing import List, Dict, Tuple

from utils import Visual
from utils.RemoveBedFilter import RemoveBedFilter

"""
读取原始文件的方法
"""


def read_nib(fpath):
    """
    读取.nii或.hdr文件，生成ndarray
    :param fpath: .nii文件路径
    :return: 返回numpy数组，四维，float32
    """
    img = nib.load(fpath)
    data = img.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
    # print(data.shape)
    return data


def read_raw(fpath, shape=None):
    data = np.fromfile(fpath, dtype=np.float32)
    if shape is None:
        img = data.reshape(512, 512, -1, 1)
    else:
        img = data.reshape(shape)
    return img


"""
数据集处理中，读取的操作
"""


def normalize(img, img_type):
    if img_type == "ct":
        img = (img + 1024) / 4096
    elif img_type == "pet":
        pass
    elif img_type == "dosemap":
        pass

    return img


def load(fpath, img_type):
    """
    在数据集中, 用于将文件名转换为数据
    :param fpath: 待读取文件的文件名
    :param img_type: 数据类型(ct, pet, dosemap)
    :return: 读取并处理后的
    """
    # 从文件名读取npy数据,输入的fpath是tensor,要利用.numpy()函数提取他的值
    img = np.load(fpath.numpy())

    img = normalize(img, img_type)

    return img


"""
病人相关操作
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


class PatientProcessor(object):
    """
    用于对一位患者的图像数据进行处理
    """
    project_path = r"E:\internal_dosimetry"

    def __init__(self, ID):
        # 患者ID和患者文件夹路径
        self.ID = ID
        self.patient_folder = PatientProcessor.project_path + r"\dataset\patient" + str(ID)

        # 生成的patch的个数
        self.n_patch = None

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

    # 图像信息分析
    def check_patch(self, n: int):
        ct = np.load(self.patient_folder+"\\patch\\ct\\"+str(n)+".npy")
        pet = np.load(self.patient_folder+"\\patch\\pet\\"+str(n)+".npy")
        dm = np.load(self.patient_folder+"\\patch\\dosemap_F18\\"+str(n)+".npy")

        sitk.Show(sitk.GetImageFromArray(ct), "ct")
        sitk.Show(sitk.GetImageFromArray(pet), "pet")
        sitk.Show(sitk.GetImageFromArray(dm), "dosemap")
        pass

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
        # time.sleep(0.01)
        with tqdm.tqdm(index_array) as bar:
            for i, j, k in bar:
                bar.set_description("Saving .npy file")
                n += 1
                np.save(os.path.join(folder_path_ct, str(n) + ".npy"), ct[i:i + size, j:j + size, k:k + size])
                np.save(os.path.join(folder_path_pet, str(n) + ".npy"), pet[i:i + size, j:j + size, k:k + size])
                np.save(os.path.join(folder_path_dm, str(n) + ".npy"), dm[i:i + size, j:j + size, k:k + size])

    @staticmethod
    def _source_tensor(particle, energy):
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
            n = 6
            quit("Particle type error!")
        source = np.zeros(shape=[8, 8, 8, 5])
        source[:, :, :, n] = energy
        return tf.constant(source, dtype=tf.float32)

    def create_train_dataset(self, particle, energy):
        """
        创建一个患者的所有patch的数据集,
        其中: ct, pet, dosemap的patch数据需提前生产; source的输入在函数中生成.
        :param particle: 生成dosemap的粒子的类型
        :param energy: 粒子的能量
        :return: ds, 结构为zip(ct, pet, source, dosemap)
        """
        # 创建文件名的数据集
        ct = tf.data.Dataset.list_files(os.path.join(self.patient_folder, "patch/ct/*.npy"), shuffle=False)
        pet = tf.data.Dataset.list_files(os.path.join(self.patient_folder, "patch/pet/*.npy"), shuffle=False)
        dosemap = tf.data.Dataset.list_files(os.path.join(self.patient_folder, "patch/dosemap/*.npy"), shuffle=False)
        # 计算患者patch的个数
        self.n_patch = len(list(ct.as_numpy_iterator()))
        # 利用load函数加载数据集
        ct = ct.map(lambda x: tf.py_function(func=load, inp=[x, "ct"], Tout=tf.float32))
        pet = pet.map(lambda x: tf.py_function(func=load, inp=[x, "pet"], Tout=tf.float32))
        dosemap = dosemap.map(lambda x: tf.py_function(func=load, inp=[x, "dosemap"], Tout=tf.float32))

        # self._source_tensor("positron", 10)
        source = tf.data.Dataset.from_tensors(self._source_tensor(particle, energy)).repeat(self.n_patch)

        ds = tf.data.Dataset.zip((ct, pet, source, dosemap))

        return ds


def create_train_dataset(p_ids, batch):
    ds = None
    n_patch = 0

    # 将所有病人的dataset连接起来
    for i, p_id in enumerate(p_ids):
        patient = PatientProcessor(ID=p_id)
        if i == 0:
            ds = patient.create_train_dataset(particle="positron", energy=0.2498)
        else:
            ds = ds.concatenate(patient.create_train_dataset(particle="positron", energy=0.2498))
        n_patch += patient.n_patch

    # ds = ds.shuffle(buffer_size=n_patch).batch(batch)
    ds = ds.batch(batch)

    return ds


if __name__ == '__main__':
    p = PatientProcessor(1)
    # p.create_npy_origin(recreate=True)
    # p.create_ct_without_bed()
    # p.create_dosemap_pet_without_air(pet="pet_origin", dm="dm_origin", reference_image="ct_AirRemoved", method='MaskedCT')
    # p.create_patches(ct="ct_AirRemoved", pet="pet_AirRemoved", dm="dm_AirRemoved", recreate_index_array=False)
    p.check_patch(2)



