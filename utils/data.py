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
        self.patient_folder = "dataset/patient" + ID

        self.ct_origin = None
        self.pet_origin = None
        self.dosemap_origin = None
        self.atlas_origin = None

        self.ct = None
        self.pet = None
        self.dosemap = None
        self.atlas = None

    def load_origin(self):
        self.ct_origin = nib.load(os.path.join(self.patient_folder, "ct.hdr"))
        self.pet_origin = nib.load(os.path.join(self.patient_folder, "pet.hdr"))
        self.atlas_origin = nib.load(os.path.join(self.patient_folder, "atlas.hdr"))

    def load_ndarray(self):
        self.ct = np.load(os.path.join(self.patient_folder, "ct.npy"))
        self.pet = np.load(os.path.join(self.patient_folder, "pet.npy"))
        self.dosemap = np.load(os.path.join(self.patient_folder, "dosemap.npy"))
        self.atlas = np.load(os.path.join(self.patient_folder, "atlas.npy"))

    def create_ndarray(self):
        self.load_origin()
        self.ct = self.ct_origin.get_fdata().squeeze(4).squeeze(4).astype(np.float32)
        self.pet = self.pet_origin.get_fdata().squeeze(4).squeeze(4).astype(np.float32)

        self.dosemap = read_raw(os.path.join(self.patient_folder, "dosemap.raw"), self.ct.shape)

        np.save(os.path.join(self.patient_folder, "ct"), self.ct)
        np.save(os.path.join(self.patient_folder, "pet"), self.pet)
        np.save(os.path.join(self.patient_folder, "dosemap"), self.dosemap)



