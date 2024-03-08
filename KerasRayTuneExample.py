# def objective(x, a, b):
#     return a * (x ** 0.5) + b
#
# import time
# from ray import tune
# from ray.air import session
# from ray.air.checkpoint import Checkpoint
#
# def train_func(config):
#     step = 0
#     loaded_checkpoint = session.get_checkpoint()
#     if loaded_checkpoint:
#         last_step = loaded_checkpoint.to_dict()["step"]
#         step = last_step + 1
#
#     for iter in range(step, 150):
#         time.sleep(1)
#
#         checkpoint = Checkpoint.from_dict({"step": step})
#         session.report({"message": "Hello world Ray Tune!"}, checkpoint=checkpoint)
#
# tuner = tune.Tuner(train_func)
# results = tuner.fit()



# from ray import tune
# from ray.tune.search.hyperopt import HyperOptSearch
# from tensorflow.python import keras
# from tensorflow.python.keras.layers import Dense
# from keras.layers import Dense, Dropout, Activation
# from tensorflow.python.keras.utils.np_utils import to_categorical
# import numpy as np
#
# x_train = np.random.random((1000, 20))
# y_train = to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# x_test = np.random.random((100, 20))
# y_test = to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#
#
# # 1. Wrap a Keras model in an objective function.
# def objective(config):
#     model = keras.models.Sequential()
#     model.add(Dense(784, activation=config["activation"]))
#     model.add(Dense(10, activation="softmax"))
#
#     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#     model.fit(x_train, y_train)
#     loss, accuracy = model.evaluate(x_test, y_test)
#     return {"accuracy": accuracy}
#
#
# # 2. Define a search space and initialize the search algorithm.
# search_space = {"activation": tune.choice(["relu", "tanh"])}
# algo = HyperOptSearch()
#
# # 3. Start a Tune run that maximizes accuracy.
# tuner = tune.Tuner(
#     objective,
#     tune_config=tune.TuneConfig(
#         metric="accuracy",
#         mode="max",
#         search_alg=algo,
#     ),
#     param_space=search_space,
# )
# results = tuner.fit()


# !/usr/bin/env python
# coding: utf-8
#
# This example showcases how to use TF2.0 APIs with Tune.
# Original code: https://www.tensorflow.org/tutorials/quickstart/advanced
#
# As of 10/12/2019: One caveat of using TF2.0 is that TF AutoGraph
# functionality does not interact nicely with Ray actors. One way to get around
# this is to `import tensorflow` inside the Tune Trainable.
#




import argparse
import os
from tensorflow.keras.datasets import mnist
from filelock import FileLock
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
# from ray.tune.search.hyperopt import HyperOptSearch
import keras


# # 1. Wrap a Keras model in an objective function.
# def objective(config):
#     model = keras.models.Sequential()
#     model.add(keras.layers.Dense(784, activation=config["activation"]))
#     model.add(keras.layers.Dense(10, activation="softmax"))
#
#     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#     # model.fit(...)
#     # loss, accuracy = model.evaluate(...)
#     return {"accuracy": accuracy}

def train_mnist(config):
    # https://github.com/tensorflow/tensorflow/issues/32159
    import tensorflow as tf
    batch_size = 128
    num_classes = 10
    epochs = 12

    with FileLock(os.path.expanduser("~/.data.lock")):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(config["hidden"], activation=config["activation"]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(
            lr=config["lr"], momentum=config["momentum"]),
        metrics=["accuracy"])

    loss, accuracy= model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReportCallback({
            "mean_accuracy": "accuracy"
        })])
    return {"accuracy": accuracy}

# 2. Define a search space and initialize the search algorithm.
search_space = {"activation": tune.choice(["relu", "tanh"]),
                "lr": tune.uniform(0.001, 0.1),
                "momentum": tune.uniform(0.1, 0.9),
                "hidden": tune.randint(32, 512)}
algo = AsyncHyperBandScheduler()

# 3. Start a Tune run that maximizes accuracy.
tuner = tune.Tuner(
    train_mnist,
    tune_config=tune.TuneConfig(
        metric="accuracy",
        mode="max",
        search_alg=algo,
    ),
    param_space=search_space,
)
results = tuner.fit()