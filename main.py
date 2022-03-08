import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os

from utils import Data, Visual
from model import AutoEncoder


"""
    前期数据生成**********************************************************************************************************
"""


# 删除文件
def remove_file(fpath):

    if os.path.exists(fpath):
        os.remove(fpath)
        print(fpath, " removed")
    else:
        print(fpath, " do not exist")


def remove_patient_file(IDs, fpath_relative):
    for ID in IDs:
        fpath = "dataset/patient" + str(ID) + "/" + fpath_relative
        remove_file(fpath)


# remove_patient_file([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "hdr/ct_noBed.hdr")
# remove_patient_file([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "hdr/ct_noBed.img")

# for ID in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
#    remove_file("dataset/patient"+str(ID)+"/nii/Patient"+str(ID)+"_CT_noBed.nii")


# 移除CT中的床位信息
def remove_bed(IDs, lower_threshold=600):
    for ID in IDs:
        read_path = "dataset/patient" + str(ID) + "/nii/Patient" + str(ID) + "_CT.nii"
        save_path = "dataset/patient" + str(ID) + "/nii/ct_AutoRemoveBed.nii"
        RemoveBed.remove_bed(read_path, save_path, lower_threshold)
        print("Finish patient ", ID)


# remove_bed([1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
# remove_bed([1, 2])


# 从hdr文件生成npy文件
def create_npy(IDs, isDm=True, isOther=True):
    for ID in IDs:
        patient = data.PatientProcessor(ID)
        patient.create_npy(isDm, isOther)
        print("Finish patient ", ID)


# create_npy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], isDm=False)
# create_npy([1, 2, 4, 5, 6])
# create_npy([3, 7, 8])


def create_ct_without_bed(IDs):
    for ID in IDs:
        patient = data.PatientProcessor(ID)
        patient.create_ct_without_bed()
        print("Finish patient ", ID)


# create_ct_without_bed([1, 2, 3, 4, 5, 6, 7, 8])

'''ct = np.load("dataset/patient1/npy/ct.npy")
ct_img = sitk.GetImageFromArray(ct)
sitk.WriteImage(ct_img, "dataset/test_ct.nii")'''


def create_dosemap_without_air(IDs):
    for ID in IDs:
        patient = data.PatientProcessor(ID)
        patient.create_dosemap_pet_without_air()
        print("Finish patient ", ID)


# create_dosemap_without_air([1, 2, 3, 4, 5, 6, 7, 8])
'''dosemap = np.load("dataset/patient1/dosemap_F18/dosemap.npy")
dosemap_img = sitk.GetImageFromArray(dosemap)
sitk.WriteImage(dosemap_img, "dataset/test_dosemap.nii")'''


# 生成切块文件
def create_patch(IDs):
    for ID in IDs:
        patient = data.PatientProcessor(ID)
        patient.create_patch_pro()
        print("Finish patient ", ID)


# create_patch([1])


"""
    数据分析*************************************************************************************************************
"""

# ct = Visual.Image("dataset/patient2/npy/ct.npy", "ct")
# ct.hist("dataset/patient2/")


def hist(IDs):
    for ID in IDs:
        patient = data.PatientProcessor(ID)
        patient.hist()
        print("Finish patient ", ID)


# hist([1, 2, 3, 4, 5, 6, 7, 8])


def info_numerical(IDs, isDm=True, isOther=True):
    for ID in IDs:
        patient = data.PatientProcessor(ID)
        patient.info_numerical(isDm=isDm, isOther=isOther)
        print("Finish patient ", ID)


# info_numerical([1, 2, 3, 4, 5, 6, 7, 8])
# info_numerical([9, 10, 11], isDm=False)


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
    网络相关*************************************************************************************************************
"""


def show_net_structure():
    net = AutoEncoder.AutoEncoder1()
    print(net.summary())
    tf.keras.utils.plot_model(net, to_file="AutoEncoder1.png", show_shapes=True, dpi=600)


# show_net_structure()


def show_patient_dataset():
    p = data.PatientProcessor(5)
    p.create_train_dataset("positron", 20)


# show_patient_dataset()


def show_train_dataset():
    ds = data.create_train_dataset(p_ids=[5], batch=4)
    print(ds)
    for n, (ct, pet, source, dosemap) in ds.enumerate():
        if n == 0:
            print(ct)

# show_train_dataset()


ct_origin = np.load(r"E:\internal_dosimetry\dataset\patient1\npy\ct_origin.npy")
ct = np.load(r"E:\internal_dosimetry\dataset\patient1\npy\ct.npy")

print(ct_origin.shape)
print(ct.shape)

