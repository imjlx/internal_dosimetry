import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import os

from utils import Data, Visual
import Framework

from model import AutoEncoder


"""
    网络相关*************************************************************************************************************
"""


def show_net_structure():
    net = AutoEncoder.AutoEncoder1()
    print(net.summary())
    tf.keras.utils.plot_model(net, to_file="AutoEncoder1.png", show_shapes=True, dpi=600)


if __name__ == "__main__":
    framework = Framework.SequentialFramework(model=AutoEncoder.AutoEncoder1(),
                                              train_IDs=(1, 2, 3, 4, 5, 6, 7, 8, 9), test_IDs=(10, ), batch=4)
    framework.fit(200)


