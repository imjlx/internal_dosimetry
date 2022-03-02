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
def _create_dosemap_without_air_SimpleThresholdOnCT(dm: np.ndarray, ct: np.ndarray, threshold: int) -> np.ndarray:
    """
    设定一个最低的阈值, 低于该阈值的CT均认为是背景, 并将对应的dosemap设置为0
    :param dm: 处理的dosemap
    :param ct: 参考的ct
    :param threshold: 阈值
    :return: 处理后的dosemap
    """
    for i, pixel in enumerate(ct.flat):
        if pixel < threshold:
            dm.flat[i] = 0
    return dm


@nb.jit(parallel=True)
def _create_patch_index_array(img: np.ndarray, size: int, step: int, ratio: float) -> np.ndarray:
    """
    根据img生成可取的块的坐标
    :param img: 参考轮廓遮罩图
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
        """
        初始化
        :param ID: 患者的序号
        """

        # 患者ID和患者文件夹路径
        self.ID = ID
        self.patient_folder = PatientProcessor.project_path + r"\dataset\patient" + str(ID)

        # 保存nib读取的原始文件
        self.ct_hdr = None
        self.ct_AutoRemoveBed_hdr = None
        self.pet_hdr = None
        self.dosemap_hdr = None
        self.atlas_hdr = None

        # 保存图像文件，4维
        self.ct = None
        self.ct_origin = None
        self.ct_AutoRemoveBed = None
        self.pet = None
        self.dosemap = None
        self.dosemap_withAir = None
        self.atlas = None

        # 图像的shape
        self.shape = None

        # 生成的patch的个数
        self.n_patch = None

    # 测试中...
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

    def quicklook_npy(self, fpath):
        fpath = os.path.join(self.patient_folder, fpath)
        img = np.load(fpath)
        count = 0
        for i in img.flat:
            if i == -1024:
                count += 1
        print(count / len(img.flat))

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
    def create_dosemap_without_air(self, dm: str = "dm_origin", ct: str = "ct_origin",
                                   method: str = "SimpleThresholdOnCT", **kwargs) -> None:

        # 读取文件
        dm = self._load_npy(dm)
        ct = self._load_npy(ct)
        # 不同的处理方法
        if method == "SimpleThresholdOnCT":
            """
            设定一个最低的阈值, 低于该阈值的CT均认为是背景, 并将对应的dosemap设置为0
            """
            # 如果没有指定threshold, 则设置为900
            if 'threshold' not in kwargs.keys():
                kwargs['threshold'] = 900
            # 利用类外numba加速的函数处理
            dm = _create_dosemap_without_air_SimpleThresholdOnCT(dm=dm, ct=ct, threshold=kwargs['threshold'])
            # dm = create_dosemap_without_air(dm, ct)
        elif method == "None":
            pass

        np.save(file=os.path.join(self.patient_folder, "dosemap_F18\\dm_AirRemoved.npy"), arr=dm)
        print(f"Patient{self.ID}'s dm_AirRemoved.npy origin file created.")

    def create_ct_without_bed(self):
        self.ct_origin = np.load(os.path.join(self.patient_folder, "npy/ct_origin.npy"))
        self.ct_AutoRemoveBed = np.load(os.path.join(self.patient_folder, "npy/ct_AutoRemoveBed.npy"))
        self.atlas = np.load(os.path.join(self.patient_folder, "npy/atlas.npy"))

        self.ct = create_ct_without_bed(self.ct_origin, self.atlas, self.ct_AutoRemoveBed)

        np.save(os.path.join(self.patient_folder, "npy/ct.npy"), self.ct)

    # 图像信息分析
    def info_numerical(self, isDm=True, isOther=True):

        info = pd.read_excel("dataset/info.xlsx", index_col="ID")

        if isOther:
            self.ct = np.load(os.path.join(self.patient_folder, "npy/ct.npy"))
            self.pet = np.load(os.path.join(self.patient_folder, "npy/pet.npy"))
            self.atlas = np.load(os.path.join(self.patient_folder, "npy/atlas.npy"))

            # ct和pet处理相同
            for img_type, img in zip(["ct", "pet"], [self.ct, self.pet]):
                img_max = img.max(initial=None)
                average = np.average(img)
                median = np.median(img)
                percentile = np.percentile(img, 99.8)

                info.loc[str(self.ID) + "_" + img_type, "Min":"percentile_r"] = [
                    img.min(initial=None), img_max,
                    average, average / img_max,
                    median, median / img_max,
                    percentile, percentile / img_max
                ]

            # atlas只求最值
            info.loc[str(self.ID) + "_atlas", "Min":"Max"] = [self.atlas.max(initial=None),
                                                              self.atlas.min(initial=None)]

        if isDm:
            img_type = "dosemap"
            img = np.load(os.path.join(self.patient_folder, "dosemap_F18/dosemap.npy"))

            img_max = img.max(initial=None)
            average = np.average(img)
            median = np.median(img)
            percentile = np.percentile(img, 99.8)

            info.loc[str(self.ID) + "_" + img_type, "Min":"percentile_r"] = [
                img.min(initial=None), img_max,
                average, average / img_max,
                median, median / img_max,
                percentile, percentile / img_max
            ]

        info.to_excel("dataset/info.xlsx")

    def hist(self):
        self.load_npy()
        for data, img_type in zip([self.ct, self.ct_origin, self.ct_AutoRemoveBed,
                                   self.dosemap_withAir, self.dosemap, self.atlas, self.pet],
                                  ["ct", "ct_origin", "ct_AutoRemoveBed",
                                   "dosemap_withAir", "dosemap", "atlas", "pet"]):
            img = Visual.Image(data, img_type)
            img.hist(self.patient_folder)

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

        return index_array

    def create_patches(self, ct: str, pet: str, dm: str, reference_img: np.ndarray = None,
                       size: int = 128, step: int = 16, ratio: float = 0.5,
                       recreate_index_array: bool = False) -> None:

        # 创建保存patch的文件夹
        folder_path_ct = self.patient_folder + "/patch/ct"
        folder_path_pet = self.patient_folder + "/patch/pet"
        folder_path_dm = self.patient_folder + "/patch/dosemap_F18"
        for dirname in [folder_path_ct, folder_path_pet, folder_path_dm]:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        # 设置参考轮廓的默认值--分割图像
        if reference_img is None:
            reference_img = self._load_npy("atlas_origin")
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

    # 已替代, 存档用
    def create_npy(self, isDm=True, isOther=True):
        """
        从hdr文件生成npy文件
        :param isDm: 是否生成dosemap
        :param isOther: 是否生成CT, PET, atlas
        :return: 无
        """
        # 判断保存路径是否存在, 若不存在则创建
        dirname = self.patient_folder + "/npy"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # 读取原始文件, 生成npy文件
        self.load_hdr(isDm=isDm, isOther=isOther)

        if isOther:
            self.ct_origin = self.ct_hdr.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
            self.ct_AutoRemoveBed = self.ct_AutoRemoveBed_hdr.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
            self.pet = self.pet_hdr.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
            self.atlas = self.atlas_hdr.get_fdata().squeeze(4).squeeze(4).astype(np.uint8)
            np.save(os.path.join(self.patient_folder, "npy/ct_origin.npy"), self.ct_origin)
            np.save(os.path.join(self.patient_folder, "npy/ct_AutoRemoveBed.npy"), self.ct_AutoRemoveBed)
            np.save(os.path.join(self.patient_folder, "npy/pet.npy"), self.pet)
            np.save(os.path.join(self.patient_folder, "npy/atlas.npy"), self.atlas)

        if isDm:
            self.dosemap_withAir = self.dosemap_hdr.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
            np.save(os.path.join(self.patient_folder, "dosemap_F18/dosemap_withAir.npy"), self.dosemap_withAir)

    def load_hdr(self, isDm=True, isOther=True):
        """
        用nibabel读取.hdr & .img文件，读取后的结果为nib中的类，保存在self.XX_origin中
        """
        if isOther:
            self.ct_hdr = nib.load(os.path.join(self.patient_folder, "hdr/ct.hdr"))
            self.ct_AutoRemoveBed_hdr = nib.load(os.path.join(self.patient_folder, "hdr/ct_AutoRemoveBed.hdr"))
            self.pet_hdr = nib.load(os.path.join(self.patient_folder, "hdr/pet.hdr"))
            self.atlas_hdr = nib.load(os.path.join(self.patient_folder, "hdr/atlas.hdr"))

        if isDm:
            self.dosemap_hdr = nib.load(os.path.join(self.patient_folder, "dosemap_F18/dosemap.hdr"))

        # 将基本信息保存在excel中
        info = pd.read_excel("dataset/info.xlsx", index_col="ID")

        if isOther:
            for img_type, header in zip(["ct", "atlas", "pet"],
                                        [self.ct_hdr.header, self.atlas_hdr.header, self.pet_hdr.header]):
                info.loc[str(self.ID) + "_" + img_type, "dimx":"pixdimz"] = [
                    header.get_data_shape()[0], header.get_data_shape()[1], header.get_data_shape()[2],
                    header.get_zooms()[0], header.get_zooms()[1], header.get_zooms()[2]]
        if isDm:
            img_type = "dosemap"
            header = self.dosemap_hdr.header
            info.loc[str(self.ID) + "_" + img_type, "dimx":"pixdimz"] = [
                header.get_data_shape()[0], header.get_data_shape()[1], header.get_data_shape()[2],
                header.get_zooms()[0], header.get_zooms()[1], header.get_zooms()[2]]

        info.to_excel("dataset/info.xlsx")

    def load_npy(self):
        """
        读取npy文件
        """
        self.ct = np.load(os.path.join(self.patient_folder, "npy/ct.npy"))
        self.ct_origin = np.load(os.path.join(self.patient_folder, "npy/ct_origin.npy"))
        self.ct_AutoRemoveBed = np.load(os.path.join(self.patient_folder, "npy/ct_AutoRemoveBed.npy"))
        self.pet = np.load(os.path.join(self.patient_folder, "npy/pet.npy"))
        self.dosemap = np.load(os.path.join(self.patient_folder, "dosemap_F18/dosemap.npy"))
        self.dosemap_withAir = np.load(os.path.join(self.patient_folder, "dosemap_F18/dosemap_withAir.npy"))
        self.atlas = np.load(os.path.join(self.patient_folder, "npy/atlas.npy"))

        self.shape = self.ct.shape


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
    p.create_patches(ct="ct_origin", pet="pet_origin", dm="dm_AirRemoved", recreate_index_array=True)
