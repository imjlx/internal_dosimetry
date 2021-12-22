import numpy as np
import os
import nibabel as nib
import tensorflow as tf
import numba as nb
import tqdm
import time

import dataset

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
def patch_index(atlas, size=128, step=16, ratio=0.5):
    """
    根据分割数据生成可取的块的坐标
    :param atlas: 分割数据,ndarray
    :param size: 块的大小
    :param step: 块的间距
    :param ratio: 是否取的依据: 组织的占比
    :return: 可取的坐标的list
    """
    # 计算每个维度可取的个数
    n_i = int(np.floor((atlas.shape[0] - size) / step) + 1)
    n_j = int(np.floor((atlas.shape[1] - size) / step) + 1)
    n_k = int(np.floor((atlas.shape[2] - size) / step) + 1)

    # patch的位置
    count_index = []

    # 对所有的起始点进行遍历
    for i in range(n_i):
        i *= step
        for j in range(n_j):
            j *= step
            for k in range(n_k):
                k *= step

                # 此时i,j,k均为块的开始坐标
                # 据此切割出atlas
                atlas_patch = atlas[i:i + size, j:j + size, k:k + size, :]
                # 判断
                if np.sum(atlas_patch > 0) / atlas_patch.size > ratio:
                    count_index.append([i, j, k])
    return count_index


class Patient(object):
    def __init__(self, ID):
        # 患者ID和患者文件夹路径
        self.ID = ID
        self.patient_folder = "dataset/patient" + str(ID)

        # 保存nib读取的原始文件
        self.ct_origin = None
        self.pet_origin = None
        self.dosemap_origin = None
        self.atlas_origin = None

        # 保存图像文件，4维
        self.ct = None
        self.pet = None
        self.dosemap = None
        self.atlas = None

        # 图像的shape
        self.shape = None

        # 生成的patch的个数
        self.n_patch = None

    def load_origin(self):
        """
        用nibabel读取.hdr & .img文件，读取后的结果为nib中的类，保存在self.XX_origin中
        """
        self.ct_origin = nib.load(os.path.join(self.patient_folder, "hdr/ct.hdr"))
        self.pet_origin = nib.load(os.path.join(self.patient_folder, "hdr/pet.hdr"))
        self.dosemap_origin = nib.load(os.path.join(self.patient_folder, "dosemap_F18/dosemap.hdr"))
        self.atlas_origin = nib.load(os.path.join(self.patient_folder, "hdr/atlas.hdr"))

    def load_ndarray(self):
        """
        读取npy文件
        """
        self.ct = np.load(os.path.join(self.patient_folder, "npy/ct.npy"))
        self.pet = np.load(os.path.join(self.patient_folder, "npy/pet.npy"))
        self.dosemap = np.load(os.path.join(self.patient_folder, "dosemap_F18/dosemap.npy"))
        self.atlas = np.load(os.path.join(self.patient_folder, "npy/atlas.npy"))
        self.shape = self.ct.shape

    def create_ndarray(self):
        """
        从原始文件生成npy文件
        """
        self.load_origin()
        self.ct = self.ct_origin.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
        self.pet = self.pet_origin.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
        self.dosemap = self.dosemap_origin.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
        self.atlas = self.atlas_origin.get_fdata().squeeze(4).squeeze(4).astype(np.uint8)

        np.save(os.path.join(self.patient_folder, "npy/ct.npy"), self.ct)
        np.save(os.path.join(self.patient_folder, "npy/pet.npy"), self.pet)
        np.save(os.path.join(self.patient_folder, "dosemap_F18/dosemap.npy"), self.dosemap)
        np.save(os.path.join(self.patient_folder, "npy/atlas.npy"), self.atlas)

    def create_patch(self, size=128, step=16, ratio=0.5):
        self.load_ndarray()
        # 计算每个维度可取的个数
        n_i = np.floor((self.shape[0] - size) / step) + 1
        n_j = np.floor((self.shape[1] - size) / step) + 1
        n_k = np.floor((self.shape[2] - size) / step) + 1

        # 记录个数
        count = 0
        count_map = np.zeros((n_i.astype(np.uint8), n_j.astype(np.uint8), n_k.astype(np.uint8)), dtype=np.uint8)

        # 对所有的起始点进行遍历
        for i in np.arange(0, step * n_i, step, dtype=np.uint16):
            for j in np.arange(0, step * n_j, step, dtype=np.uint16):
                for k in np.arange(0, step * n_k, step, dtype=np.uint16):
                    ct = self.ct[i:i + size, j:j + size, k:k + size, :]
                    pet = self.pet[i:i + size, j:j + size, k:k + size, :]
                    dosemap = self.dosemap[i:i + size, j:j + size, k:k + size, :]
                    atlas = self.atlas[i:i + size, j:j + size, k:k + size, :]
                    print("(%d, %d, %d), " % (i, j, k), end=" ")

                    if self._if_count(atlas, ratio):
                        count += 1
                        count_map[int(i / step), int(j / step), int(k / step)] = 1
                        np.save(self.patient_folder + "/patch/ct/" + str(count) + ".npy", ct)
                        np.save(self.patient_folder + "/patch/pet/" + str(count) + ".npy", pet)
                        np.save(self.patient_folder + "/patch/dosemap/" + str(count) + ".npy", dosemap)
                        # np.save(self.patient_folder + "/patch/atlas/" + str(count) + ".npy", atlas)
                        print("count")
                    else:
                        print("discount")

        np.save(self.patient_folder + "/patch/count_map.npy", count_map)
        print(count)

    def create_patch_pro(self, size=128, step=16, ratio=0.5, refresh=False):

        # 判断保存路径是否存在, 若不存在则创建
        dirname_ct = self.patient_folder + "/patch/ct"
        dirname_pet = self.patient_folder + "/patch/pet"
        dirname_dosemap = self.patient_folder + "/patch/dosemap_F18"
        for dirname in [dirname_ct, dirname_pet, dirname_dosemap]:
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        # 加载数据
        self.load_ndarray()

        # 获取可取patch的坐标
        # 坐标文件的路径
        index_path = self.patient_folder+"/patch/index_list.npy"
        # 如果坐标文件不存在或刷新, 生成新的坐标文件
        if (not os.path.exists(index_path)) or refresh:
            # 生成可取块的坐标
            print("Generating Index....")
            t_start = time.time()
            index_list = patch_index(self.atlas, size=size, step=step, ratio=ratio)
            t_end = time.time()
            print("Index Generated! %d patches, spent %.2f s" % (len(index_list), t_end - t_start))
            # 保存坐标
            index_list = np.array(index_list)
            np.save(self.patient_folder + "/patch/index_list.npy", index_list)
        # 如果文件存在且不刷新, 直接读取
        else:
            index_list = np.load(index_path)
            print("Index loaded! %d patches" % len(index_list))



        # 提取并保存为npy文件
        n = 0
        time.sleep(0.01)
        with tqdm.tqdm(index_list) as bar:
            for i, j, k in bar:
                bar.set_description("Saving .npy file")

                n += 1
                np.save(os.path.join(dirname_ct, str(n)+".npy"), self.ct[i:i + size, j:j + size, k:k + size, :])
                np.save(os.path.join(dirname_pet, str(n)+".npy"), self.pet[i:i + size, j:j + size, k:k + size, :])
                np.save(os.path.join(dirname_dosemap, str(n)+".npy"), self.dosemap[i:i + size, j:j + size, k:k + size, :])


    @staticmethod
    def _if_count(atlas, ratio):
        is_organ = 0
        for i in range(atlas.shape[0]):
            for j in range(atlas.shape[1]):
                for k in range(atlas.shape[2]):
                    if atlas[i, j, k, 0] != 0:
                        is_organ += 1
        if is_organ > ratio * atlas.size:
            return True
        else:
            return False

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
        patient = Patient(ID=p_id)
        if i == 0:
            ds = patient.create_train_dataset(particle="positron", energy=0.2498)
        else:
            ds = ds.concatenate(patient.create_train_dataset(particle="positron", energy=0.2498))
        n_patch += patient.n_patch

    # ds = ds.shuffle(buffer_size=n_patch).batch(batch)
    ds = ds.batch(batch)

    return ds
