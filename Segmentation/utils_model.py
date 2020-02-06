import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K


def block_down(inputs, filters, drop, w_decay=0.0001, kernel_size=3):
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      kernel_regularizer=regularizers.l2(w_decay), activation="elu")(inputs)
    c = layers.Conv2D(filters, (kernel_size, kernel_size), activation='elu', kernel_initializer='he_normal',
                      padding='same',
                      kernel_regularizer=regularizers.l2(w_decay))(x)
    p = layers.MaxPooling2D((2, 2))(c)
    p = layers.Dropout(drop)(p)
    return c, p


def bridge(inputs, filters, drop, kernel_size=3):
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      activation="elu")(inputs)
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      activation="elu")(x)
    x = layers.Dropout(drop)(x)
    return x


def block_up(inputs, conc, filters, drop, w_decay=0.0001, kernel_size=3):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same',
                               kernel_regularizer=regularizers.l2(w_decay))(
        inputs)
    for i in range(len(conc)):
        x = layers.concatenate([x, conc[i]])
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      kernel_regularizer=regularizers.l2(w_decay), activation="elu")(x)
    x = layers.Conv2D(filters, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same',
                      kernel_regularizer=regularizers.l2(w_decay), activation="elu")(x)
    x = layers.Dropout(drop)(x)
    return x


