# -*- coding: utf-8 -*-
"""IDL_A1_Task1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MvX5ZP1LQuzlXsjf3GSBNATFYE8mAIKp
"""

### Task 1

#Setting seed for reproducibility and check TensorFlow version
import tensorflow as tf

tf.keras.utils.set_random_seed(3311791)
tf.__version__

#Plain mnist_mlp.py from https://github.com/keras-team/keras/blob/tf-keras-2/examples/mnist_mlp.py#L22
'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Visualization of accuracy, loss and architecture
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import plot_model
def visualize(model, history, epochs=20, filename='model.png', title=None):
    tf.keras.backend.clear_session() # reset layer counter

    pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, epochs-1], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
    plt.title(title)
    plt.show()
    return plot_model(model, filename, show_shapes=True, show_layer_names=True)

visualize(model, history, title='MNIST MLP from Keras Repository')

#Experimenting with the MLP: remove Dropout layers
def mlp_keras(num_dense=512, epochs=20, batch_size=128, title=None):
    keras.backend.clear_session() # reset layer name count
    model = Sequential()
    model.add(Dense(num_dense, activation='relu', input_shape=(784,)))
    model.add(Dense(num_dense, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
 #   visualize(model, history, title=title)
    return model, history
model2, history2 = mlp_keras() # Dropout layers removed
model3, history3 = mlp_keras(num_dense=100) # Neurons in Dense hidden layers reduced
model4, history4 = mlp_keras(batch_size=16) # Batch size reduced

#Plain mnist_cnn.py from https://github.com/keras-team/keras/blob/tf-keras-2/examples/mnist_cnn.py
# low accuracy across different GPUs etc. (code outdated)
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Experimenting with the official Keras CNN (outdated)
def cnn_keras(epochs=12, first_kernel_size=3, second_kernel_size=2, padding='valid', title=None):
    keras.backend.clear_session() # reset layer name count
    model = Sequential()
    model.add(Conv2D(32, kernel_size=first_kernel_size, padding=padding,
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, second_kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu')) # kernel_initializer="he_normal"
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
#    visualize(model, history)
    return model, history
model5, history5 = cnn_keras() # default
model6, history6 = cnn_keras(padding='same')
model7, history7 = cnn_keras(first_kernel_size=5, second_kernel_size=3, padding='same')
# signigicantly less test accuracy for all models than what was promised

#demo_mnist_convnet.py from official keras repo (only 2 months since last update)
# https://github.com/keras-team/keras/blob/master/examples/demo_mnist_convnet.py
import numpy as np
import keras
from keras import layers
from keras.utils import to_categorical

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

batch_size = 128
epochs = 3

model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

model.fit(
    x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#Experimenting with the official up-to-date Keras CNN
def cnn_keras(epochs=3, batch_size=128, first_kernel_size=3, second_kernel_size=3,
                  padding='valid', optimizer="adam", add_dropout=None, add_dense=None,
                  max_pooling2d=(2, 2), title=None):
    keras.backend.clear_session() # reset layer name count

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=first_kernel_size, activation="relu", padding=padding),
        ]
    )

    if max_pooling2d is not None:
        model.add(layers.MaxPooling2D(pool_size=max_pooling2d))

    model.add(layers.Conv2D(64, kernel_size=second_kernel_size, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    if add_dropout is not None:
        model.add(Dropout(add_dropout))

    model.add(layers.Flatten())

    if add_dense is not None:
        model.add(Dense(add_dense, activation="relu"))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1
    )

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
 #   visualize(model, history, epochs=epochs)
    return model, history

model5, history5 = cnn_keras() # updated keras model
model6, history6 = cnn_keras(padding='same')
model62, history62 = cnn_keras(padding='same', max_pooling2d=None)
model7, history7 = cnn_keras(first_kernel_size=5, second_kernel_size=3, padding='same')
model8, history8 = cnn_keras(add_dropout=0.25, add_dense=128, optimizer=keras.optimizers.Adadelta(), max_pooling2d=None)
model9, history9 = cnn_keras(add_dense=128, optimizer=keras.optimizers.Adadelta(), max_pooling2d=None)
model10, history10 = cnn_keras(add_dense=128, optimizer=keras.optimizers.Adadelta(), max_pooling2d=None)
model11, history11 = cnn_keras(optimizer=keras.optimizers.Adadelta())

visualize(model8, history8, epochs=3)

## Part 2
# MLP

import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-6000], y_train_full[:-6000]
X_valid, y_valid = X_train_full[-6000:], y_train_full[-6000:] # 10% of training set as validation

print(X_train.shape, X_test.shape, X_valid.shape)
print(y_train.shape, y_test.shape, y_valid.shape)

#Scale the pixel intensities down to the 0-1 range and convert them to floats, by dividing by 255
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

#Display sample of Fashion MNIST images
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
n_rows = 3
n_cols = 8
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.savefig("fashion_mnist_sample.png")
plt.show()

keras.backend.clear_session()
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
               optimizer="sgd",
               metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=30,
                validation_data=(X_valid, y_valid))
history.params
model.evaluate(X_test, y_test)

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q -U keras_tuner

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import regularizers

def build_mlp(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam", "Adadelta", "nadam"])
    activation = hp.Choice("activation", values=["relu", "tanh", "elu"])
    initializer = hp.Choice("initializer", values=["glorot_uniform", "he_uniform"])
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    l1_reg = hp.Float("l1_reg", min_value=0.0, max_value=0.1, step=0.01)
    l2_reg = hp.Float("l2_reg", min_value=0.0, max_value=0.1, step=0.01)

    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer == "Adadelta":
        optimizer = "Adadelta"
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation=activation, kernel_initializer=initializer,
                                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model

random_search_tuner = kt.RandomSearch(
    build_mlp, objective="val_accuracy", max_trials=10, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)
random_search_tuner.search(X_train, y_train, epochs=10,
                           validation_data=(X_valid, y_valid))

top3_models = random_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]

#best_model.fit(X_train, y_train, epochs=30)

top3_params = random_search_tuner.get_best_hyperparameters(num_trials=3)
top3_params[0].values

print(top3_params[1].values)
print(top3_params[2].values)

#top 3 hypertuned models
top3_trials = random_search_tuner.oracle.get_best_trials(num_trials=3)
print(top3_trials[0].metrics.get_last_value("val_accuracy"))
print(top3_trials[1].metrics.get_last_value("val_accuracy"))
print(top3_trials[2].metrics.get_last_value("val_accuracy"))

## MLP CIFAR-10 test

import keras
from keras.datasets import cifar10

# Load the CIFAR-10 dataset
(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

n_rows = 3
n_cols = 8
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))

for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train_cifar[index], cmap="viridis", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train_cifar[index][0]])

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.savefig("cifar_10_sample.png")
plt.show()

X_train_cifar, X_test_cifar = X_train_cifar / 255, X_test_cifar / 255
print("X_train shape:", X_train_cifar.shape)
print(X_train_cifar.shape[0], "train samples")
print(X_test_cifar.shape[0], "test samples")

print(X_train_cifar.shape, X_test_cifar.shape)
print(y_train_cifar.shape, y_test_cifar.shape)

import tensorflow as tf
from tensorflow.keras import regularizers

def mlp_cifar(parameters):
    if parameters["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=parameters["learning_rate"])
    elif parameters["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=parameters["learning_rate"])
    elif parameters["optimizer"] == "nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=parameters["learning_rate"])
    else:
        optimizer = "Adadelta"

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))

    for _ in range(parameters['n_hidden']):
        model.add(tf.keras.layers.Dense(parameters['n_neurons'], activation=parameters['activation'], kernel_initializer=parameters['initializer'],
                                        kernel_regularizer=regularizers.l1_l2(l1=parameters['l1_reg'], l2=parameters['l2_reg'])))
        model.add(tf.keras.layers.Dropout(parameters['dropout_rate']))

    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


mlp1 = mlp_cifar(top3_params[0].values)
mlp2 = mlp_cifar(top3_params[1].values)
mlp3 = mlp_cifar(top3_params[2].values)

mlp1.fit(X_train_cifar, y_train_cifar, epochs=10)
test_loss1, test_accuracy1 = mlp1.evaluate(X_test_cifar, y_test_cifar)
mlp2.fit(X_train_cifar, y_train_cifar, epochs=10)
test_loss2, test_accuracy2 = mlp2.evaluate(X_test_cifar, y_test_cifar)
mlp3.fit(X_train_cifar, y_train_cifar, epochs=10)
test_loss3, test_accuracy3 = mlp3.evaluate(X_test_cifar, y_test_cifar)

print(test_accuracy1, test_accuracy2, test_accuracy3)

# CNN

## Baseline model
from functools import partial

keras.backend.clear_session()
DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plot_model(model, "geron_cnn.png", show_shapes=True, show_layer_names=True)

# Build model for tuning

from functools import partial
def build_cnn(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam", "Adadelta", "nadam"])
    activation = hp.Choice("activation", values=["relu", "tanh", "elu"])
    initializer = hp.Choice("initializer", values=["glorot_uniform", "he_uniform"])
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    l1_reg = hp.Float("l1_reg", min_value=0.0, max_value=0.1, step=0.01)
    l2_reg = hp.Float("l2_reg", min_value=0.0, max_value=0.1, step=0.01)

    if optimizer == "sgd": # adadelta has no learning rate hyperparameter
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

    model = keras.models.Sequential()

    model.add(DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=128))
    model.add(DefaultConv2D(filters=128))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=256))
    model.add(DefaultConv2D(filters=256))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation=activation, kernel_initializer=initializer,
                                        kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(units=10, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model

random_search_tuner = kt.RandomSearch(
    build_cnn, objective="val_accuracy", max_trials=10, overwrite=True,
    directory="my_fashion_mnist2", project_name="my_rnd_search2", seed=42)
random_search_tuner.search(X_train, y_train, epochs=10,
                           validation_data=(X_valid, y_valid))

top3_cnns = random_search_tuner.get_best_models(num_models=3)
best_cnn = top3_cnns[0]

#top 3 hypertuned models
top3_trials_cnn = random_search_tuner.oracle.get_best_trials(num_trials=3)
print(top3_trials_cnn[0].metrics.get_last_value("val_accuracy"))
print(top3_trials_cnn[1].metrics.get_last_value("val_accuracy"))
print(top3_trials_cnn[2].metrics.get_last_value("val_accuracy"))

best_cnns = random_search_tuner.get_best_hyperparameters(num_trials=3)
best_cnns[0].values

print(best_cnns[1].values)
print(best_cnns[2].values)

def cnn_cifar(params):
    if params["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=params["learning_rate"])
    elif params["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    elif params["optimizer"] == "nadam":
        optimizer = tf.keras.optimizers.Nadam(learning_rate=params["learning_rate"])
    else:
        optimizer = "Adadelta"
    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")

    model = keras.models.Sequential()

    model.add(DefaultConv2D(filters=64, kernel_size=7, input_shape=[32, 32, 3]))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=128))
    model.add(DefaultConv2D(filters=128))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(DefaultConv2D(filters=256))
    model.add(DefaultConv2D(filters=256))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())

    for _ in range(params['n_hidden']):
        model.add(tf.keras.layers.Dense(params['n_neurons'], activation=params['activation'], kernel_initializer=params['initializer'],
                                        kernel_regularizer=regularizers.l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))
        model.add(tf.keras.layers.Dropout(params['dropout_rate']))

    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


    return model

cnn1 = cnn_cifar(best_cnns[0].values)
cnn2 = cnn_cifar(best_cnns[1].values)
cnn3 = cnn_cifar(best_cnns[2].values)

cnn1.fit(X_train_cifar, y_train_cifar, epochs=10)
test_loss4, test_accuracy4 = cnn1.evaluate(X_test_cifar, y_test_cifar)

cnn2.fit(X_train_cifar, y_train_cifar, epochs=10)
test_loss5, test_accuracy5 = cnn2.evaluate(X_test_cifar, y_test_cifar)

cnn3.fit(X_train_cifar, y_train_cifar, epochs=10)
test_loss6, test_accuracy6 = cnn3.evaluate(X_test_cifar, y_test_cifar)

print(test_accuracy4, test_accuracy5, test_accuracy6)