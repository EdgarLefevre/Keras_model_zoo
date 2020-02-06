#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.backend as K
import tensorflow as tf
from .losses import soft_n_cut_loss
import numpy as np


def block_down(inputs, filters, drop=0, w_decay=0.0001, kernel_size=3):
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      kernel_regularizer=regularizers.l2(w_decay), activation="elu")(inputs)
    c = layers.Conv2D(filters, (kernel_size, kernel_size), activation='elu', kernel_initializer='he_normal',
                      padding='same',
                      kernel_regularizer=regularizers.l2(w_decay))(x)
    p = layers.MaxPooling2D((2, 2))(c)
    p = layers.Dropout(drop)(p)
    return p, c


def bridge(inputs, filters, drop=0, kernel_size=3):
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      activation="elu")(inputs)
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      activation="elu")(x)
    x = layers.Dropout(drop)(x)
    return x


def block_up(input, conc, filters, drop=0, w_decay=0.0001, kernel_size=3):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same',
                               kernel_regularizer=regularizers.l2(w_decay))(input)
    for i in range(len(conc)):
        x = layers.concatenate([x, conc[i]])
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      kernel_regularizer=regularizers.l2(w_decay), activation="elu")(x)
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      kernel_regularizer=regularizers.l2(w_decay), activation="elu")(x)
    x = layers.Dropout(drop)(x)
    return x


def unet(input_size, enc, name):
    input = layers.Input(input_size)
    # down
    # d1, c1 = block_down(input, filters=64)
    d2, c2 = block_down(input, filters=64)
    d3, c3 = block_down(d2, filters=128)
    d4, c4 = block_down(d3, filters=256)

    # bridge
    b = bridge(d4, filters=512)

    # up
    u4 = block_up(input=b, filters=256, conc=[c4])
    u3 = block_up(input=u4, filters=128, conc=[c3])
    u2 = block_up(input=u3, filters=64, conc=[c2])
    # u1 = block_up(input=u2, filters=64, conc=[c1])
    if enc:
        output = layers.Conv2D(1, (1, 1))(u2)
        output = layers.Dense(units=1, activation="softmax")(output)  # todo : 1 ou 2 ici ??
    else:
        output = layers.Conv2D(1, (1, 1))(u2)
    return keras.Model(input, output, name=name)


def u_enc(input_size):
    return unet(input_size, True, name="encoder")


def u_dec(input_size):
    return unet(input_size, False, name="decoder")


def wnet(input_shape, lr):
    input = layers.Input(input_shape)
    uenc = u_enc(input_shape)
    x = uenc(input)
    udec = u_dec((x.shape[1], x.shape[2], x.shape[3]))
    output = udec(x)

    model = keras.Model(inputs=input, outputs=output, name="wnet")

    # reconstruction_loss = tf.pow(K.flatten(input) - K.flatten(output), 2)
    reconstruction_loss = keras.metrics.binary_crossentropy(K.flatten(input),
                                                            K.flatten(output))  # todo : use l2 loss here
    soft_ncut_loss = soft_n_cut_loss(flatten_image=K.flatten(input), prob=x,  # todo : prob = row*col*k tensor
                                     k=1, rows=input_shape[0],
                                     cols=input_shape[1])
    final_loss = reconstruction_loss + soft_ncut_loss

    model.add_loss(final_loss)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    return model

