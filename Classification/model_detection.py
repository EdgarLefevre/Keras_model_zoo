
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def model_detection(input_shape, lr, n_classes=2):
    inputs = layers.Input(input_shape)

    b1_c = layers.Conv2D(32, (3, 3), activation="linear", padding='same')(inputs)
    b1_c = layers.Dropout(rate=0.5)(b1_c)  # initially 0.4
    b1_l = layers.LeakyReLU(alpha=0.1)(b1_c)
    b1_mp = layers.MaxPool2D(2)(b1_l)

    b2_c = layers.Conv2D(64, (3, 3), activation="linear", padding='same')(b1_mp)
    b2_c = layers.Dropout(rate=0.5)(b2_c)
    b2_l = layers.LeakyReLU(alpha=0.1)(b2_c)
    b2_mp = layers.MaxPool2D(2)(b2_l)

    b3_c = layers.Conv2D(128, (3, 3), activation="linear", padding='same')(b2_mp)
    b3_c = layers.Dropout(rate=0.5)(b3_c)
    b3_l = layers.LeakyReLU(alpha=0.1)(b3_c)
    b3_mp = layers.MaxPool2D(2)(b3_l)

    b4_c = layers.Conv2D(256, (3, 3), activation="linear", padding='same')(b3_mp)
    b4_c = layers.Dropout(rate=0.5)(b4_c)
    b4_l = layers.LeakyReLU(alpha=0.1)(b4_c)
    b4_mp = layers.MaxPool2D(2)(b4_l)

    flat = layers.Flatten()(b4_mp)

    b4_d = layers.Dense(128, activation='linear')(flat)
    b4_l = layers.LeakyReLU(alpha=0.1)(b4_d)

    outputs = layers.Dense(n_classes, activation='softmax')(b4_l)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['accuracy'])

    return model
