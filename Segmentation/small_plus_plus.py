
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from . import utils_model


def small_plus_plus(input_shape, filters=16):
    inputs = layers.Input(input_shape)

    c2, p2 = utils_model.block_down(inputs, filters=filters, drop=0.1)
    c3, p3 = utils_model.block_down(p2, filters=2 * filters, drop=0.2)
    c4, p4 = utils_model.block_down(p3, filters=4 * filters, drop=0.2)

    o = utils_model.bridge(p4, filters=8 * filters, drop=0.3)

    u4 = utils_model.block_up(o, [c4], filters=4 * filters, drop=0.2)

    n3_1 = utils_model.block_up(c4, [c3], filters=2 * filters, drop=0.2)
    u3 = utils_model.block_up(u4, [n3_1, c3], filters=2 * filters, drop=0.2)

    n2_1 = utils_model.block_up(c3, [c2], filters=filters, drop=0.1)
    n2_2 = utils_model.block_up(n3_1, [n2_1, c2], filters=filters, drop=0.1)
    u2 = utils_model.block_up(u3, [n2_2, n2_1, c2], filters=filters, drop=0.1)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(u2)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    return model
