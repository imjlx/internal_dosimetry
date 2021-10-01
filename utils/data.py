import numpy as np
import os
import nibabel as nib


def header_info(fpath):
    img = nib.load(fpath)
    print(img.header)
    return img


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


def save_ndarray(dtype, r_fpath, s_fpath):
    if dtype == "nii":
        img = read_nib(r_fpath)
        np.save(s_fpath, img)
    elif dtype == "hdr":
        pass


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

    def load_origin(self):
        self.ct_origin = nib.load(os.path.join(self.patient_folder, "ct.hdr"))
        self.pet_origin = nib.load(os.path.join(self.patient_folder, "pet.hdr"))
        self.dosemap_origin = nib.load(os.path.join(self.patient_folder, "dosemap.hdr"))
        self.atlas_origin = nib.load(os.path.join(self.patient_folder, "atlas.hdr"))

    def load_ndarray(self):
        self.ct = np.load(os.path.join(self.patient_folder, "ct.npy"))
        self.pet = np.load(os.path.join(self.patient_folder, "pet.npy"))
        self.dosemap = np.load(os.path.join(self.patient_folder, "dosemap.npy"))
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
        np.save(os.path.join(self.patient_folder, "dosemap.npy"), self.dosemap)
        np.save(os.path.join(self.patient_folder, "atlas.npy"), self.atlas)

    def create_block(self, size=128, step=16, ratio=0.5):
        self.load_ndarray()
        # 计算每个维度可取起始点的个数
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







