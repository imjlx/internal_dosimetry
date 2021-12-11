from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers

DROPOUT_RATE = 0.5
INITIALIZER = tf.random_normal_initializer(0.0, 0.02)


def conv3x3x3(filters):
    return layers.Conv3D(filters=filters, kernel_size=3, strides=1, padding='same',
                         use_bias=False, kernel_initializer=INITIALIZER)


class Conv3x3x3(tf.keras.Model, ABC):
    def __init__(self, out_filters):
        super().__init__()
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv = conv3x3x3(filters=out_filters)
        self.dropout = layers.Dropout(rate=DROPOUT_RATE)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        return x


def down_stack(inputs, out_filters):
    x = Conv3x3x3(out_filters=out_filters/2)(inputs)
    x = Conv3x3x3(out_filters=out_filters/2)(x)
    outputs = layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x)
    return outputs


def up_stack(inputs, out_filters):
    x = Conv3x3x3(out_filters=out_filters*2)(inputs)
    x = Conv3x3x3(out_filters=out_filters*2)(x)
    outputs = layers.UpSampling3D(size=2)(x)
    return outputs


def AutoEncoder1():
    ct_inputs = layers.Input(shape=[128, 128, 128, 1], name="CT")
    pet_inputs = layers.Input(shape=[128, 128, 128, 1], name="PET")

    ct = down_stack(ct_inputs, 32)  # 64
    ct = down_stack(ct, 64)  # 32
    ct = down_stack(ct, 128)  # 16
    ct = down_stack(ct, 256)  # 8

    # pet = layers.Concatenate()([pet_inputs, energy_inputs])
    pet = down_stack(pet_inputs, 32)  # 64
    pet = down_stack(pet, 64)  # 32
    pet = down_stack(pet, 128)  # 16
    pet = down_stack(pet, 256)  # 8

    source_inputs = layers.Input(shape=[8, 8, 8, 5], name="source")
    x = layers.Concatenate()([ct, pet, source_inputs])

    x = up_stack(x, 128)
    x = up_stack(x, 64)
    x = up_stack(x, 32)
    x = up_stack(x, 16)
    dosemap = Conv3x3x3(1)(x)

    return tf.keras.Model(inputs=[ct_inputs, pet_inputs, source_inputs], outputs=dosemap)











