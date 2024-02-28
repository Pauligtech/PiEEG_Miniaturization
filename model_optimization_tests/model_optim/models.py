import os
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
from itertools import chain
from functools import partial

# JAX + Keras
os.environ["KERAS_BACKEND"] = "jax"
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import jax
import jax.numpy as jnp
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
)
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from keras.constraints import max_norm
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, BatchNormalization

NUM_CLASSES = None
NUM_CHANNELS = None
NUM_SAMPLES = None


# region Helper funcs
def shallow_conv_net_square_layer(x):
    return jnp.square(x)


def shallow_conv_net_log_layer(x):
    return jnp.log(jnp.clip(x, 1e-7, 10000))


CUSTOM_OBJECTS = {
    "shallow_conv_net_square_layer": shallow_conv_net_square_layer,
    "shallow_conv_net_log_layer": shallow_conv_net_log_layer,
}
# endregion Helper funcs


# region Models
def eeg_net(
    nb_classes=NUM_CLASSES,
    channels=NUM_CHANNELS,
    samples=NUM_SAMPLES,
    dropout_rate=0.5,
    kernLength=64,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout",
):
    """
    From: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    """

    dropoutType = {"Dropout": Dropout, "SpatialDropout2D": SpatialDropout2D}[
        dropoutType
    ]

    input1 = Input(shape=(channels, samples, 1))

    ##################################################################
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(channels, samples, 1),
        use_bias=False,
    )(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        (channels, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropout_rate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropout_rate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)

    return Model(inputs=input1, outputs=softmax)


def deep_conv_net(
    nb_classes=NUM_CLASSES, channels=NUM_CHANNELS, samples=NUM_SAMPLES, dropout_rate=0.5
):
    """
    From: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    """
    input_main = Input((channels, samples, 1))
    block1 = Conv2D(
        25,
        (1, 5),
        input_shape=(channels, samples, 1),
        kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
    )(input_main)
    block1 = Conv2D(25, (channels, 1), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(
        block1
    )
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation("elu")(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropout_rate)(block1)

    block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2 = Activation("elu")(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropout_rate)(block2)

    block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(
        block2
    )
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3 = Activation("elu")(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropout_rate)(block3)

    block4 = Conv2D(200, (1, 5), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(
        block3
    )
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4 = Activation("elu")(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropout_rate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation("softmax")(dense)

    return Model(inputs=input_main, outputs=softmax)


def shallow_conv_net(
    nb_classes=NUM_CLASSES, channels=NUM_CHANNELS, samples=NUM_SAMPLES, **kwargs
):
    """
    From: https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
    """

    _POOL_SIZE_D2_ = kwargs.get("pool_size_d2", 35)
    _STRIDES_D2_ = kwargs.get("strides_d2", 7)
    _CONV_FILTERS_D2_ = kwargs.get("conv_filters_d2", 13)

    _POOL_SIZE_ = kwargs.get("pool_size", (1, _POOL_SIZE_D2_))
    _STRIDES_ = kwargs.get("strides", (1, _STRIDES_D2_))
    _CONV_FILTERS_ = kwargs.get("conv_filters", (1, _CONV_FILTERS_D2_))

    _CONV2D_1_UNITS_ = kwargs.get("conv2d_1_units", 40)
    _CONV2D_2_UNITS_ = kwargs.get("conv2d_2_units", 40)
    _L2_REG_1_ = kwargs.get("l2_reg_1", 0.01)
    _L2_REG_2_ = kwargs.get("l2_reg_2", 0.01)
    _L2_REG_3_ = kwargs.get("l2_reg_3", 0.01)
    _DROPOUT_RATE_ = kwargs.get("dropout_rate", 0.5)

    input_main = Input(shape=(channels, samples, 1))
    block1 = Conv2D(
        _CONV2D_1_UNITS_,
        _CONV_FILTERS_,
        input_shape=(channels, samples, 1),
        kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        kernel_regularizer=keras.regularizers.L2(_L2_REG_1_),
    )(input_main)
    # block1       = Conv2D(40, (channels, 1), use_bias=False,
    #                       kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1 = Conv2D(
        _CONV2D_2_UNITS_,
        (channels, 1),
        use_bias=False,
        kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        kernel_regularizer=keras.regularizers.L2(_L2_REG_2_),
    )(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(shallow_conv_net_square_layer)(block1)
    block1 = AveragePooling2D(pool_size=_POOL_SIZE_, strides=_STRIDES_)(block1)
    block1 = Activation(shallow_conv_net_log_layer)(block1)
    block1 = Dropout(_DROPOUT_RATE_)(block1)
    flatten = Flatten()(block1)
    # dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    dense = Dense(
        nb_classes,
        kernel_constraint=max_norm(0.5),
        kernel_regularizer=keras.regularizers.L2(_L2_REG_3_),
    )(flatten)
    softmax = Activation("softmax")(dense)

    return Model(inputs=input_main, outputs=softmax)


def lstm_net(nb_classes=NUM_CLASSES, channels=NUM_CHANNELS, samples=NUM_SAMPLES):
    model = Sequential()
    model.add(Input(shape=(samples, channels)))
    model.add(LSTM(40, return_sequences=True, stateful=False))
    # model.add(LSTM(40, return_sequences=True, stateful=False, batch_input_shape=(batch_size, timesteps, data_dim)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(40, return_sequences=True, stateful=False))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(LSTM(40, stateful=False))
    model.add(Dropout(0.5))
    # model.add(TimeDistributed(Dense(T_train)))
    model.add(Dense(50, activation="softmax"))
    model.add(Dense(nb_classes, activation="softmax"))

    return model


def lstm_cnn_net(
    nb_classes=NUM_CLASSES, channels=NUM_CHANNELS, samples=NUM_SAMPLES, **kwargs
):

    _CONV1D_1_UNITS_ = kwargs.get("conv1d_1_units", 40)
    _CONV1D_1_KERNEL_SIZE_ = kwargs.get("conv1d_1_kernel_size", 20)
    _CONV1D_1_STRIDES_ = kwargs.get("conv1d_1_strides", 4)
    _CONV1D_1_MAXPOOL_SIZE_ = kwargs.get("conv1d_1_maxpool_size", 4)
    _CONV1D_1_MAXPOOL_STRIDES_ = kwargs.get("conv1d_1_maxpool_strides", 4)
    _LSTM_1_UNITS_ = kwargs.get("lstm_1_units", 50)
    _L2_REG_1_ = kwargs.get("l2_reg_1", 0.01)
    _L2_REG_2_ = kwargs.get("l2_reg_2", 0.01)
    _L2_REG_3_ = kwargs.get("l2_reg_3", 0.01)
    _DROPOUT_RATE_1_ = kwargs.get("dropout_rate_1", 0.5)
    _DROPOUT_RATE_2_ = kwargs.get("dropout_rate_2", 0.5)

    model = Sequential()
    model.add(Input(shape=(samples, channels)))

    # add 1-layer cnn
    model.add(
        Conv1D(
            _CONV1D_1_UNITS_,
            kernel_size=_CONV1D_1_KERNEL_SIZE_,
            strides=_CONV1D_1_STRIDES_,
            kernel_regularizer=keras.regularizers.L2(_L2_REG_1_),
        )
    )
    model.add(Activation("relu"))
    model.add(Dropout(_DROPOUT_RATE_1_))
    model.add(BatchNormalization())
    model.add(
        MaxPooling1D(
            pool_size=_CONV1D_1_MAXPOOL_SIZE_, strides=_CONV1D_1_MAXPOOL_STRIDES_
        )
    )

    # add 1-layer lstm
    model.add(
        LSTM(
            _LSTM_1_UNITS_,
            return_sequences=True,
            stateful=False,
            kernel_regularizer=keras.regularizers.L2(_L2_REG_2_),
        )
    )
    model.add(Dropout(_DROPOUT_RATE_2_))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(
        Dense(
            nb_classes,
            activation="softmax",
            kernel_regularizer=keras.regularizers.L2(_L2_REG_3_),
        )
    )

    return model


def lstm_cnn_net_v2(nb_classes=NUM_CLASSES, channels=NUM_CHANNELS, samples=NUM_SAMPLES):
    model = Sequential()
    model.add(Input(shape=(samples, channels)))

    # Add 1-layer LSTM
    model.add(LSTM(50, return_sequences=True, stateful=False))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    # Add 1-layer CNN
    model.add(Conv1D(40, kernel_size=20, strides=4))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=4, strides=4))
    model.add(Flatten())
    # Fully connected layer for classification
    model.add(Dense(nb_classes, activation="softmax"))
    return model


# endregion Models
