import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from utils import data, visual


def create_block(ID):
    patient = data.Patient(ID)
    patient.create_block()


def show_data_distribution(fpath="dataset/patient5/pet.npy", img_type='ct'):
    img = np.load(fpath)
    print("shape: ", img.shape)
    print("dtype: ", img.dtype)
    print("max:   ", img.max())
    print("min:   ", img.min())
    # img_per = img.clip(min=100)
    percentile = np.percentile(img, 99.8)
    print(percentile)
    arr = img.flatten()
    plt.figure()
    plt.hist(arr, bins=200, range=(percentile/50, percentile))
    plt.xlim(0, percentile)
    plt.title("histogram")
    plt.show()


# show_data_distribution(fpath="dataset/patient5/atlas.npy", img_type="atlas")


def show_img(fpath="dataset/patient5/patch/pet/1.npy", img_type="ct", x=64, y=64, z=64):
    img = np.load(fpath)
    visual.imshow3D(img, img_type, x, y, z)


show_img(fpath="dataset/patient5/atlas.npy", img_type="atlas", x=256, y=256, z=500)

