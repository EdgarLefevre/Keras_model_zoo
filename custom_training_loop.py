import os
import time

import tensorflow as tf
import tensorflow.keras as keras


@tf.function
def train_step(model, x, y, loss_fn, optimizer, train_metric):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss_value = loss_fn(y, pred)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_metric.update_state(y, pred)


@tf.function
def test_step(model, x, y, test_metric):
    val_logits = model(x, training=False)
    test_metric.update_state(y, val_logits)


def train_model(model, epochs, train_dataset, val_dataset, lr):
    train_acc_metric = keras.metrics.BinaryAccuracy()
    val_acc_metric = keras.metrics.BinaryAccuracy()
    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(lr=lr)
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            train_step(
                model,
                x_batch_train,
                y_batch_train,
                loss_fn,
                optimizer,
                train_acc_metric,
            )
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_acc_metric.reset_states()
        for x_batch_val, y_batch_val in val_dataset:
            test_step(model, x_batch_val, y_batch_val, val_acc_metric)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
