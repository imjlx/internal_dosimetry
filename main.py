import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import dataset
from utils import data, Visual
from model import AutoEncoder


def create_npy(ID):
    patient = data.Patient(ID)
    patient.create_ndarray()


# create_npy(6)


def create_patch(ID):
    patient = data.Patient(ID)
    patient.create_patch_pro()


create_patch(5)


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
