import tensorflow as tf

def weighted_cross_entropy(y_true, y_pred):
    """
    -- Fonction de coût pondéré --
    :param y_true: vrai valeur de y (label)
    :param y_pred: valeur prédite de y par le modèle
    :return: valeur de la fonction de cout d'entropie croisée pondérée
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=3)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    epsilon = tf.convert_to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = tf.zeros_like(y_pred, dtype=y_pred.dtype)  # array_ops
    cond = (y_pred >= zeros)
    relu_logits = tf.where(cond, y_pred, zeros)
    neg_abs_logits = tf.where(cond, -y_pred, y_pred)
    entropy = tf.math.add(relu_logits - y_pred * seg,
                          tf.math.log1p(
                              tf.math.exp(neg_abs_logits)),
                          name=None)

    return K.mean(tf.multiply(weight, entropy), axis=-1)


# @tf.function
# def mean_iou(y_true, y_pred):
#     """
#     :param y_true: array de label annote.
#     :param y_pred: array de label predit par le modele.
#     :return: valeur de l'IoU.
#     """
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         # m = tf.keras.metrics.MeanIoU(num_classes=2)
#         # m.update_state(y_true, y_pred)
#         # score = m.result()
#         # print(score)
#         score, up_opt = tf.metrics.MeanIoU(y_true, y_pred, 2)
#         tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.local_variables_initializer())  # todo : fix this line
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)


def mean_iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = (K.sum(y_true_f + y_pred_f)) - intersection
    return intersection / union


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def soft_dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def loss_maison(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(y_true, y_pred)
