import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import dataset
from utils import data, Visual
from model import AutoEncoder
from utils import RemoveBed


"""
    前期数据生成
"""


# 移除CT中的床位信息
def remove_bed(IDs):
    for ID in IDs:
        read_path = "dataset/patient" + str(ID) + "/nii/Patient" + str(ID) + "_CT.nii"
        save_path = "dataset/patient" + str(ID) + "/nii/Patient" + str(ID) + "_CT_noBed.nii"
        RemoveBed.remove_bed(read_path, save_path)
        print("Finish patient ", ID)


# remove_bed([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
remove_bed([11])


# 从hdr文件生成npy文件
def create_npy(IDs):
    for ID in IDs:
        patient = data.Patient(ID)
        patient.create_ndarray()
        print("Finish patient ", ID)


# create_npy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


# 生成切块文件
def create_patch(IDs):
    for ID in IDs:
        patient = data.Patient(ID)
        patient.create_patch_pro()
        print("Finish patient ", ID)


# create_patch([1])


"""
    数据分析
"""


def show_data_distribution(fpath, cut):
    img = np.load(fpath)
    print("shape: ", img.shape)
    print("dtype: ", img.dtype)
    print("min:   ", img.min())
    print("max:   ", img.max())
    # print("SUM:   ", img.sum())

    arr = img.flatten()
    plt.figure()

    if cut is True:
        percentile = np.percentile(img, 99.9)
        print(percentile)
        plt.hist(arr, bins=200, range=(percentile/50, percentile))
        plt.xlim(0, percentile)
    else:
        plt.hist(arr, bins=200)

    plt.title("histogram")
    plt.show()


# show_data_distribution(fpath="dataset/patient1/dosemap_F18/dosemap.npy", cut=False)
# show_data_distribution(fpath="dataset/patient1/pet.npy", cut=True)


def show_img(fpath, img_type="ct", x=None, y=None, z=None):
    img = np.load(fpath)
    Visual.imshow3D(img, img_type, x, y, z)


# show_img(fpath="dataset/patient5/patch/ct/1.npy", img_type="ct", x=None, y=None, z=None)
# show_img(fpath="dataset/patient5/patch/pet/1.npy", img_type="pet", x=None, y=None, z=None)
# show_img(fpath="dataset/patient5/patch/dosemap/1.npy", img_type="dosemap", x=None, y=None, z=None)
# show_img(fpath="dataset/patient6/ct.npy", img_type="ct", x=None, y=None, z=None)
# show_img(fpath="dataset/patient6/pet.npy", img_type="pet", x=None, y=None, z=None)
# show_img(fpath="dataset/patient6/dosemap_F18/dosemap.npy", img_type="dosemap", x=None, y=None, z=None)



"""
    网络相关
"""


def show_net_structure():
    net = AutoEncoder.AutoEncoder1()
    print(net.summary())
    tf.keras.utils.plot_model(net, to_file="AutoEncoder1.png", show_shapes=True, dpi=600)


# show_net_structure()


def show_patient_dataset():
    p = data.Patient(5)
    p.create_train_dataset("positron", 20)


# show_patient_dataset()


def show_train_dataset():
    ds = data.create_train_dataset(p_ids=[5], batch=4)
    print(ds)
    for n, (ct, pet, source, dosemap) in ds.enumerate():
        if n == 0:
            print(ct)

# show_train_dataset()
