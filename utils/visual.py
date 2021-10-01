import matplotlib.pyplot as plt
import numpy as np
import cv2


def imshow3D(img, x, y, z):
    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(img[x, :, :, 0], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(img[:, y, :, 0], cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(img[:, :, z, 0], cmap="gray")
    plt.show()


def imshow3D_opencv(img, img_type, x=256, y=256, z=600):
    """

    :param img:
    :param img_type:
    :param x:
    :param y:
    :param z:
    :return:
    """
    # 首先将输入归一化至（0，1），不同的输入归一化取值范围不一样
    if img_type == "ct":
        img = (img + 1024)/4096
    elif img_type == "dosemap":
        img = np.clip(img * 10**8, 0, 1)
    cv2.imshow("x", img[x, :, :, 0].T)
    cv2.imshow("y", img[:, y, :, 0].T)
    cv2.imshow("z", img[:, :, z, 0].T)
    cv2.waitKey()


def histogram(img, img_type):
    array = img.flatten()
    plt.figure()
    if img_type == "dosemap":
        plt.hist(array, range=(0, 0.2 * 10**(-8)), bins=256, facecolor='green', alpha=0.75)
        plt.xlim(0, 0.2 * 10**(-8))
        plt.ylim(0, 0.1 * 10**8)
    elif img_type == "atlas":
        plt.hist(array, range=(1, 19), bins=20)
        plt.xlim(0, 19)
        # plt.ylim(0, 0.1 * 10**8)
    plt.show()
    
    
    