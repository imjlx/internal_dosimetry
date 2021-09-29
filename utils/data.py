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


def save_ndarray(dtype, r_fpath, s_fpath):
    if dtype == "nii":
        img = read_nib(r_fpath)
        np.save(s_fpath, img)
    elif dtype == "hdr":
        pass



