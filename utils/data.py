import numpy as np
import os
import nibabel as nib
import tensorflow as tf

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


def read(fpath):

    img = np.load(fpath)
    return img


def normalize(img, img_type):

    if img_type == "ct":
        img = (img + 1024) / 4096
    elif img_type == "pet":
        pass
    elif img_type == "dosemap":
        pass

    return img


def load(fpath, img_type):

    img = read(fpath.numpy())
    img = normalize(img, img_type)

    return img


"""
病人相关操作
"""


class Patient(object):
    def __init__(self, ID):
        self.ID = ID
        self.patient_folder = "dataset/patient" + str(ID)

        self.ct_origin = None
        self.pet_origin = None
        self.dosemap_origin = None
        self.atlas_origin = None

        self.ct = None
        self.pet = None
        self.dosemap = None
        self.atlas = None
        self.shape = None

        self.n_patch = None

    def load_origin(self):
        self.ct_origin = nib.load(os.path.join(self.patient_folder, "hdr/ct.hdr"))
        self.pet_origin = nib.load(os.path.join(self.patient_folder, "hdr/pet.hdr"))
        self.dosemap_origin = nib.load(os.path.join(self.patient_folder, "dosemap_F18/dosemap.hdr"))
        self.atlas_origin = nib.load(os.path.join(self.patient_folder, "hdr/atlas.hdr"))

    def load_ndarray(self):
        self.ct = np.load(os.path.join(self.patient_folder, "ct.npy"))
        self.pet = np.load(os.path.join(self.patient_folder, "pet.npy"))
        self.dosemap = np.load(os.path.join(self.patient_folder, "dosemap_F18/dosemap.npy"))
        self.atlas = np.load(os.path.join(self.patient_folder, "atlas.npy"))
        self.shape = self.ct.shape

    def create_ndarray(self):
        self.load_origin()
        self.ct = self.ct_origin.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
        self.pet = self.pet_origin.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
        self.dosemap = self.dosemap_origin.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
        self.atlas = self.atlas_origin.get_fdata().squeeze(4).squeeze(4).astype(np.uint8)

        np.save(os.path.join(self.patient_folder, "ct.npy"), self.ct)
        np.save(os.path.join(self.patient_folder, "pet.npy"), self.pet)
        np.save(os.path.join(self.patient_folder, "dosemap_F18/dosemap.npy"), self.dosemap)
        np.save(os.path.join(self.patient_folder, "atlas.npy"), self.atlas)

    def create_patch(self, size=128, step=16, ratio=0.5):
        self.load_ndarray()
        # 计算每个维度可的个数
        n_i = np.floor((self.shape[0]-size)/step) + 1
        n_j = np.floor((self.shape[1]-size)/step) + 1
        n_k = np.floor((self.shape[2]-size)/step) + 1

        # 记录个数
        count = 0
        count_map = np.zeros((n_i.astype(np.uint8), n_j.astype(np.uint8), n_k.astype(np.uint8)), dtype=np.uint8)

        # 对所有的起始点进行遍历
        for i in np.arange(0, step*n_i, step, dtype=np.uint16):
            for j in np.arange(0, step*n_j, step, dtype=np.uint16):
                for k in np.arange(0, step*n_k, step, dtype=np.uint16):
                    ct = self.ct[i:i+size, j:j+size, k:k+size, :]
                    pet = self.pet[i:i+size, j:j+size, k:k+size, :]
                    dosemap = self.dosemap[i:i+size, j:j+size, k:k+size, :]
                    atlas = self.atlas[i:i+size, j:j+size, k:k+size, :]
                    print("(%d, %d, %d), " % (i, j, k), end=" ")

                    if self._if_count(atlas, ratio):
                        count += 1
                        count_map[int(i/step), int(j/step), int(k/step)] = 1
                        np.save(self.patient_folder + "/patch/ct/" + str(count) + ".npy", ct)
                        np.save(self.patient_folder + "/patch/pet/" + str(count) + ".npy", pet)
                        np.save(self.patient_folder + "/patch/dosemap/" + str(count) + ".npy", dosemap)
                        # np.save(self.patient_folder + "/patch/atlas/" + str(count) + ".npy", atlas)
                        print("count")
                    else:
                        print("discount")

        np.save(self.patient_folder + "/patch/count_map.npy", count_map)
        print(count)

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

        ct = tf.data.Dataset.list_files(os.path.join(self.patient_folder, "patch/ct/*.npy"), shuffle=False)
        pet = tf.data.Dataset.list_files(os.path.join(self.patient_folder, "patch/pet/*.npy"), shuffle=False)
        dosemap = tf.data.Dataset.list_files(os.path.join(self.patient_folder, "patch/dosemap/*.npy"), shuffle=False)

        self.n_patch = len(list(ct.as_numpy_iterator()))

        ct = ct.map(lambda x: tf.py_function(func=load, inp=[x, "ct"], Tout=tf.float32))
        pet = pet.map(lambda x: tf.py_function(func=load, inp=[x, "pet"], Tout=tf.float32))
        dosemap = dosemap.map(lambda x: tf.py_function(func=load, inp=[x, "dosemap"], Tout=tf.float32))

        self._source_tensor("positron", 10)
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

