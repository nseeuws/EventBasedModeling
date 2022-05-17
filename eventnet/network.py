from collections.abc import Callable
import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D, ELU, BatchNormalization, MaxPool1D,
    UpSampling1D, Concatenate, Activation,
    SeparableConv1D, Dropout, Conv2D,
    MaxPool2D, UpSampling2D, Add, Multiply,
    AveragePooling2D
)
from tensorflow.keras import initializers
from tensorflow.python.ops import nn


def artefact_convolution_block(
        filters: int, kernel_size=5
) -> Callable:
    """
    Basic building block for the artefact EventNet
    :param filters: Number of filters in the convolution
    :param kernel_size: Kernel size of the convolution
    :return: The symbolic tensor representing block output
    """
    def build_block(x):
        x = SeparableConv1D(
            filters=filters, kernel_size=kernel_size,
            strides=1, padding='same', activation=None
        )(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        return x
    return build_block


def encoding_artefact_block(
        filters: int, kernel_size: int
) -> Callable:
    """
    Artefact EventNet encoder building block
    :param filters: Number of filters in the convolutions
    :param kernel_size: Kernel size of the convolutions
    :return: Symbolic tensor for the block output
    """
    def build_block(x):
        x = artefact_convolution_block(
            filters=filters, kernel_size=kernel_size
        )(x)
        x = artefact_convolution_block(
            filters=filters, kernel_size=kernel_size
        )(x)
        return x
    return build_block


def decoding_artefact_block(
        filters: int, kernel_size: int,
        dropout=0.2
) -> Callable:
    """
    Artefact EventNet decoder building block
    :param filters: Number of filters in the convolutions
    :param kernel_size: Kernel size of the convolutions
    :param dropout: Dropout rate between the convolutions.
    (Defaults to 0.2)
    :return: Symbolic tensor for the block output
    """
    def build_block(x):
        x = artefact_convolution_block(
            filters=filters, kernel_size=kernel_size
        )(x)
        x = Dropout(rate=dropout)(x)
        x = artefact_convolution_block(
            filters=filters, kernel_size=kernel_size
        )(x)
        return x
    return build_block


class BiasedConv(Conv2D):
    def __init__(
            self,
            filters: int, kernel_size=(1, 1),
            strides=(1, 1), padding='same',
            activation='sigmoid',
            bias_initializer='ones',
            kernel_initializer='zeros'
    ):
        super(Conv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides, padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )
        self.bias = None

    def build(self, input_shape):
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        # noinspection SpellCheckingInspection
        outputs = nn.bias_add(inputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class AttentionPooling(object):
    def __init__(self, filters, channels=18):
        self.filters = filters
        self.channels = channels

    def __call__(self, inputs):
        query, value = inputs

        att_q = Conv2D(
            filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
            padding='same', activation=None, use_bias=False
        )(query)
        att_k = Conv2D(
            filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
            padding='same', activation=None, use_bias=False
        )(value)

        gate = BiasedConv(filters=self.filters)(Add()([att_q, att_k]))
        att = Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1),
            padding='same', activation='sigmoid',
            kernel_initializer='ones', bias_initializer='zeros'
        )(gate)

        attention_output = Multiply()([att, value])

        return AveragePooling2D(pool_size=(1, self.channels), padding='same')(attention_output)


def build_artefact_eventnet(
        input_duration: int,
        filters=64, kernel_size=5
) -> Model:
    """
    Builds our artefact EventNet
    :param input_duration: Number of time steps in the network input.
    Can be `None` (for a network processing variable-size input)
    :param filters: (Base) number of filters in the network convolutions
    :param kernel_size: (Base) kernel size in the network convolutions
    :return: A `tf.keras.Model`, the full EventNet
    """
    stride = 4
    prior = 1e-2

    input_seq = Input(shape=(input_duration, 1))

    # Level 0
    x = Conv1D(
        filters=filters // 2, kernel_size=4 * kernel_size,
        strides=1, padding='same', activation=None
    )(input_seq)
    x = ELU()(x)  # First layer doesn't get a batch-normalization step

    # Level 1
    x = MaxPool1D(pool_size=stride)(x)
    x = encoding_artefact_block(
        filters=filters, kernel_size=4 * kernel_size
    )(x)

    # Level 2
    x = MaxPool1D(pool_size=stride)(x)
    x = encoding_artefact_block(
        filters=filters, kernel_size=3 * kernel_size
    )(x)
    lvl2 = x

    # Level 3
    x = MaxPool1D(pool_size=stride)(x)
    x = encoding_artefact_block(
        filters=filters, kernel_size=3 * kernel_size
    )(x)
    lvl3 = x

    # Level 4
    x = MaxPool1D(pool_size=stride)(x)
    x = encoding_artefact_block(
        filters=filters, kernel_size=2 * kernel_size
    )(x)
    lvl4 = x

    # Level 5
    x = MaxPool1D(pool_size=stride)(x)
    x = encoding_artefact_block(
        filters=filters, kernel_size=kernel_size
    )(x)
    lvl5 = x

    # Level 6
    x = MaxPool1D(pool_size=stride)(x)
    x = decoding_artefact_block(
        filters=filters, kernel_size=kernel_size
    )(x)

    # Up-level 5
    up5 = UpSampling1D(size=stride)(x)
    x = Concatenate(axis=-1)([lvl5, up5])
    x = decoding_artefact_block(
        filters=filters, kernel_size=kernel_size
    )(x)

    # Up-level 4
    up4 = UpSampling1D(size=stride)(x)
    x = Concatenate(axis=-1)([lvl4, up4])
    x = decoding_artefact_block(
        filters=filters, kernel_size=2 * kernel_size
    )(x)

    # Up-level 3
    up3 = UpSampling1D(size=stride)(x)
    x = Concatenate(axis=-1)([lvl3, up3])
    x = decoding_artefact_block(
        filters=filters, kernel_size=3 * kernel_size
    )(x)

    # Up-level 2
    up2 = UpSampling1D(size=stride)(x)
    x = Concatenate(axis=-1)([lvl2, up2])
    x = decoding_artefact_block(
        filters=filters, kernel_size=3 * kernel_size
    )(x)

    # Output heads
    bias_init = initializers.Constant(value=-np.log((1 - prior) / prior))
    center_logit = Conv1D(
        filters=1, kernel_size=kernel_size, strides=1, padding='same',
        activation=None, bias_initializer=bias_init
    )(x)
    center = Activation('sigmoid')(center_logit)
    duration = Conv1D(
        filters=1, kernel_size=kernel_size, strides=1,
        padding='same', activation='sigmoid'
    )(x)

    # Build model
    model = Model(input_seq, [center, duration, center_logit])

    return model


def seizure_convolution_block(
    filters: int, kernel_size: int
) -> Callable:
    """
    Seizure EventNet base block
    :param filters: Number of filters for the convolutions
    :param kernel_size: Kernel size for the convolutions
    :return: Function building the network layer
    """
    def build_block(x):
        x = Conv2D(
            filters=filters, kernel_size=(kernel_size, 1),
            strides=(1, 1), padding='same', activation=None
        )(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        return x
    return build_block


def build_seizure_eventnet(
        input_duration: int, n_channels=18,
        filters=16
) -> Model:
    """
    Build the seizure EventNet
    :param input_duration: Number of time steps in the network input.
    Can be `None` (for a network processing variable-size input)
    :param n_channels: Number of channels in the input EEG.
    Defaults to 18 (as used in the Neureka paper.)
    :param filters: Base number of filters in the network.
    :return: A `tf.keras.Model` containing the network.
    """
    prior = 1e-3
    stride = 4
    input_seq = Input(shape=(input_duration, n_channels, 1))

    # Level 0
    x = seizure_convolution_block(
        filters=filters, kernel_size=15
    )(input_seq)

    # Level 1
    x = MaxPool2D(pool_size=(stride, 1), padding='same')(x)
    x = seizure_convolution_block(
        filters=2 * filters, kernel_size=15
    )(x)

    # Level 2
    x = MaxPool2D(pool_size=(stride, 1), padding='same')(x)
    x = seizure_convolution_block(
        filters=4 * filters, kernel_size=15
    )(x)

    # Level 3
    x = MaxPool2D(pool_size=(stride, 1), padding='same')(x)
    x = seizure_convolution_block(
        filters=4 * filters, kernel_size=7
    )(x)

    # Level 4
    x = MaxPool2D(pool_size=(stride, 1), padding='same')(x)
    x = seizure_convolution_block(
        filters=8 * filters, kernel_size=3
    )(x)
    lvl4 = x

    # Level 5 (bottleneck level)
    x = MaxPool2D(pool_size=(stride, 1), padding='same')(x)
    x = seizure_convolution_block(
        filters=8 * filters, kernel_size=3
    )(x)

    # Pooling across channels
    x = MaxPool2D(pool_size=(1, n_channels), padding='same')(x)
    x = seizure_convolution_block(
        filters=4 * filters, kernel_size=3
    )(x)
    x = Dropout(rate=0.5)(x)
    x = seizure_convolution_block(
        filters=4 * filters, kernel_size=3
    )(x)

    # Up-level 4
    up4 = UpSampling2D(size=(stride, 1))(x)
    att4 = AttentionPooling(
        filters=4 * filters, channels=n_channels
    )([up4, lvl4])
    x = Concatenate(axis=-1)([up4, att4])
    x = seizure_convolution_block(
        filters=4 * filters, kernel_size=5
    )(x)
    x = seizure_convolution_block(
        filters=4 * filters, kernel_size=5
    )(x)

    # Output heads
    bias_init = initializers.Constant(value=np.log((1 - prior) / prior))
    center_logit = Conv2D(
        filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same',
        activation=None, bias_initializer=bias_init
    )(x)
    center = Activation('sigmoid')(center_logit)
    duration = Conv2D(
        filters=1, kernel_size=(1, 1), strides=(1, 1),
        padding='same', activation='sigmoid'
    )(x)

    model = Model(input_seq, [center, duration, center_logit])

    return model
