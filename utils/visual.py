import matplotlib.pyplot as plt
import numpy as np
import cv2


def normalization(img, img_type):
    cmap = None
    if img_type == "ct":
        img = (img + 1024) / 4096
        cmap = "gray"
    elif img_type == "pet":
        percentile = np.percentile(img, 99.8)
        img = (img / percentile).clip(max=1)
        cmap = "hot"
    elif img_type == "dosemap":
        percentile = np.percentile(img, 99.8)
        img = (img / percentile).clip(max=1)
        cmap = "gist_heat"
    elif img_type == "atlas":
        img = img / 19
        cmap = "tab20b"

    return img, cmap


def imshow3D(img, img_type, x, y, z):
    img, cmap = normalization(img, img_type)
    plt.figure(dpi=600)
    plt.subplot(1, 3, 1)
    plt.imshow(img[x, :, :, 0].T, cmap=cmap)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(img[:, y, :, 0].T, cmap=cmap)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(img[:, :, z, 0].T, cmap=cmap)
    plt.axis("off")

    plt.show()


def imshow3D_opencv(img, img_type, x=256, y=256, z=600):

    img, cmap = normalization(img, img_type)

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
    
    
    