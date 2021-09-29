import matplotlib.pyplot as plt
import cv2


def imshow3D(ndarray, x, y, z):
    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(ndarray[x, :, :, 0], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(ndarray[:, y, :, 0], cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(ndarray[:, :, z, 0], cmap="gray")
    plt.show()


def imshow3D_opencv(ndarray, x, y, z):

    img = (ndarray + 1024)/4096
    cv2.imshow("x", img[x, :, :, 0].T)
    cv2.imshow("y", img[:, y, :, 0].T)
    cv2.imshow("z", img[:, :, z, 0].T)
    cv2.waitKey()

