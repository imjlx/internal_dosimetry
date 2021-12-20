import matplotlib.pyplot as plt
import numpy as np
import cv2


def normalize(img, img_type):
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


def imshow3D(img, img_type, x=None, y=None, z=None):
    # 根据输入图像的类型进行画图前的初始化
    img, cmap = normalize(img, img_type)
    # 设置默认的x、y、z为图像的中间值
    if x is None:
        x = np.floor(img.shape[0]/2).astype(np.uint16)
    if y is None:
        y = np.floor(img.shape[0]/2).astype(np.uint16)
    if z is None:
        z = np.floor(img.shape[0]/2).astype(np.uint16)

    # 画图
    fig = plt.figure(dpi=600)
    # fig = plt.figure(figsize=(30, 15))

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(img[x, :, :, 0].T, cmap=cmap)
    ax1.axis("off")
    ax1.margins(0)

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(img[:, y, :, 0].T, cmap=cmap)
    ax2.axis("off")
    ax1.margins(0)

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(img[:, :, z, 0].T, cmap=cmap)
    ax3.axis("off")
    ax1.margins(0)

    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.9, wspace=0.05, hspace=0.05)
    fig.suptitle(img_type, fontsize=25)
    fig.show()


def imshow3D_opencv(img, img_type, x=256, y=256, z=600):

    img, cmap = normalize(img, img_type)

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
    plt.show()
    
    
    