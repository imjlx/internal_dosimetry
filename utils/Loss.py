import tensorflow as tf
import numpy as np


l1_loss = tf.keras.losses.MeanAbsoluteError()
l2_loss = tf.keras.losses.MeanSquaredError()


def l1_loss_func(y_true, y_pred):
    loss = tf.reduce_mean(y_true - y_pred)
    return loss


def l2_loss_func(y_true, y_pred):
    loss = tf.reduce_mean((y_true - y_pred) ** 2)
    return loss


def total_loss(y_true, y_pred):
    pass


if __name__ == "__main__":
    x = [[1, 2], [3, 4]]
    y = x + 2*np.ones_like(x)
    loss = l2_loss(x, y)
    pass
