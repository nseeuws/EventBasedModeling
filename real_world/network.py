import tensorflow as tf 
import numpy as np 

from tensorflow.keras import Model, Input 
from tensorflow.keras.layers import (
    Conv1D, ELU, BatchNormalization, MaxPool1D, UpSampling1D,
    Concatenate, Activation, SeparableConv1D, Dropout,
    Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D,
    Multiply, Add
)
from tensorflow.python.keras import (
    activations, constraints, 
    initializers, regularizers
)
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn

class BiasedConv(Conv2D):
    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
        super(Conv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=True,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)
    
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
    
        self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
        
        self.built = True
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
    # Check if the input_shape in call() is different from that in build().
    # If they are different, recreate the _convolution_op to avoid the stateful
    # behavior.
        call_input_shape = inputs.get_shape()
        outputs = inputs
    
        if self.data_format == 'channels_first':
            if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
        else:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class AttentionPooling(object):
    def __init__(self,filters, channels=18):
        self.filters = filters
        self.channels = channels
        
    def __call__(self, inputs):
        query, value = inputs
        
        att_q = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', activation=None, use_bias=False)(query)
        att_k = Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
                      padding='same', activation=None, use_bias=False)(value)
        gate = BiasedConv(filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', activation='sigmoid',
                         kernel_initializer='zeros', bias_initializer='ones')(Add()([att_q, att_k]))
        att = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),
                    padding='same', activation='sigmoid',
                    kernel_initializer='ones', bias_initializer='zeros')(gate)
        
        return AveragePooling2D(pool_size=(1, self.channels), padding='same')(Multiply()([att, value]))

def get_event_artefact(input_duration: int):
    # Network parameters
    n_filters = 64
    stride_factor = 4
    kernel_size = 5
    prior = 1e-2
    bn = True


    # Network
    def ConvBlock(filters, kernel=5):
        def build_block(x):
            x = SeparableConv1D(
                filters=filters, kernel_size=kernel, strides=1,
                padding='same', activation=None
            )(x)
            if bn:
                x = BatchNormalization()(x)
            x = ELU()(x)
            return x
        return build_block

    input_seq = Input(shape=(input_duration, 1))

    # Lvl0
    x = Conv1D(
        filters=n_filters//2, kernel_size=4*kernel_size, strides=1,
        padding='same', activation=None
    )(input_seq)
    #x = BatchNormalization()(x) # No bn yet. EEG is _highly_ non-Gaussian, this first feature extraction might also not yet be
    x = ELU()(x)

    #Lvl1
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=4*kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=4*kernel_size)(x)

    #Lvl2
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    lvl2 = x

    #Lvl3
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    lvl3 = x

    #Lvl4
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=2*kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=2*kernel_size)(x)
    lvl4 = x

    #Lvl5
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)
    lvl5 = x

    #Lvl6
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)

    # Up 5
    up5 = UpSampling1D(size=stride_factor)(x)
    x = Concatenate(axis=-1)([lvl5, up5])
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)

    # Up 4
    up4 = UpSampling1D(size=stride_factor)(x)
    x = Concatenate(axis=-1)([lvl4, up4])
    x = ConvBlock(filters=n_filters, kernel=2*kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=2*kernel_size)(x)
    
    # Up 3
    up3 = UpSampling1D(size=stride_factor)(x)
    x = Concatenate(axis=-1)([lvl3, up3])
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)

    # Up 2
    up2 = UpSampling1D(size=stride_factor)(x)
    x = Concatenate(axis=-1)([lvl2, up2])
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)

    # Output
    bias_init = tf.keras.initializers.Constant(value=-np.log((1 - prior) / prior))
    # I think this initialization comes from the RetinaNet paper? (The focal loss one)
    center_map_logit = Conv1D(filters=1, kernel_size=kernel_size, strides=1, padding='same',
                         activation=None, bias_initializer=bias_init)(x)
    center_map = Activation('sigmoid')(center_map_logit)
    size_map = Conv1D(filters=1, kernel_size=kernel_size, strides=1,
                padding='same', activation='sigmoid')(x)

    model = Model(input_seq, [center_map, size_map, center_map_logit])

    return model


def get_epoch_artefact(input_duration: int):
    # Network parameters
    n_filters = 64
    stride_factor = 4
    kernel_size = 5
    prior = 1e-2
    bn = True

    def ConvBlock(filters, kernel=5):
        def build_block(x):
            x = SeparableConv1D(
                filters=filters, kernel_size=kernel, strides=1,
                padding='same', activation=None
            )(x)
            if bn:
                x = BatchNormalization()(x)
            x = ELU()(x)
            return x
        return build_block

    input_seq = Input(shape=(input_duration, 1))

    # Lvl0
    x = Conv1D(
        filters=n_filters//2, kernel_size=4*kernel_size, strides=1,
        padding='same', activation=None
    )(input_seq)
    #x = BatchNormalization()(x) # No bn yet. EEG is _highly_ non-Gaussian, this first feature extraction might also not yet be
    x = ELU()(x)

    #Lvl1
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=4*kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=4*kernel_size)(x)

    #Lvl2
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    lvl2 = x

    #Lvl3
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    lvl3 = x

    #Lvl4
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=2*kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=2*kernel_size)(x)
    lvl4 = x

    #Lvl5
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)
    lvl5 = x

    #Lvl6
    x = MaxPool1D(pool_size=stride_factor)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)

    # Up 5
    up5 = UpSampling1D(size=stride_factor)(x)
    x = Concatenate(axis=-1)([lvl5, up5])
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=kernel_size)(x)

    # Up 4
    up4 = UpSampling1D(size=stride_factor)(x)
    x = Concatenate(axis=-1)([lvl4, up4])
    x = ConvBlock(filters=n_filters, kernel=2*kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=2*kernel_size)(x)
    
    # Up 3
    up3 = UpSampling1D(size=stride_factor)(x)
    x = Concatenate(axis=-1)([lvl3, up3])
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)

    # Up 2
    up2 = UpSampling1D(size=stride_factor)(x)
    x = Concatenate(axis=-1)([lvl2, up2])
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)
    x = Dropout(rate=0.2)(x)
    x = ConvBlock(filters=n_filters, kernel=3*kernel_size)(x)

    # Output
    bias_init = tf.keras.initializers.Constant(value=-np.log((1 - prior) / prior))
    # I think this initialization comes from the RetinaNet paper? (The focal loss one)
    center_map_logit = Conv1D(filters=1, kernel_size=kernel_size, strides=1, padding='same',
                         activation=None, bias_initializer=bias_init)(x)
    center_map = Activation('sigmoid')(center_map_logit)

    unet = Model(input_seq, center_map)

    return unet

def get_event_seizure(
        input_duration, n_channels = 18,
        filters=16, prior=0.001
):
    # Input convolutions
    input_seq = Input(shape=(input_duration, n_channels, 1))
    
    x = Conv2D(
        filters=filters, kernel_size=(15, 1), strides=(1, 1),
        padding='same', activation=None
    )(input_seq)
    x = BatchNormalization()(x)
    lvl0 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl0)
    x = Conv2D(
        filters=2*filters, kernel_size=(15, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl1 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl1)
    x = Conv2D(
        filters=4*filters, kernel_size=(15, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl2 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl2)
    x = Conv2D(
        filters=4*filters, kernel_size=(7, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl3 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl3)
    x = Conv2D(
        filters=8*filters, kernel_size=(3, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl4 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl4)
    x = Conv2D(
        filters=8*filters, kernel_size=(3, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl5 = ELU()(x)

    x = MaxPooling2D(pool_size=(1, n_channels), padding='same')(lvl5)
    x = Conv2D(
        filters=4*filters, kernel_size=(3, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(
        filters=4*filters, kernel_size=(3, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(rate=0.5)(x)

    up4 = UpSampling2D(size=(4, 1))(x)
    att4 = AttentionPooling(
        filters=4*filters, channels=n_channels
    )([up4, lvl4])
    x = Concatenate(axis=-1)([up4, att4])
    x = Conv2D(
        filters=4*filters, kernel_size=(5, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(
        filters=4*filters, kernel_size=(5, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    # Build outputs
    bias_init = tf.keras.initializers.Constant(value=np.log((1 - prior) / prior))
    center_map_logit = Conv2D(
        filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same',
        activation=None, bias_initializer=bias_init
    )(x)
    center_map = Activation('sigmoid')(center_map_logit)
    size_map = Conv2D(
        filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same',
        activation='sigmoid'
    )(x)

    model = Model(input_seq, [center_map, size_map, center_map_logit])
    return model


def get_epoch_seizure(input_duration: int):
    filters = 8
    n_channels = 18

    input_seq = Input(shape=(input_duration, n_channels, 1))

    x = Conv2D(
        filters=filters, kernel_size=(15, 1), strides=(1, 1),
        padding='same', activation=None
    )(input_seq)
    x = BatchNormalization()(x)
    lvl0 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl0)
    x = Conv2D(
        filters=2*filters, kernel_size=(15, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl1 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl1)
    x = Conv2D(
        filters=4*filters, kernel_size=(15, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl2 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl2)
    x = Conv2D(
        filters=4*filters, kernel_size=(7, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl3 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl3)
    x = Conv2D(
        filters=8*filters, kernel_size=(3, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl4 = ELU()(x)

    x = MaxPooling2D(pool_size=(4, 1), padding='same')(lvl4)
    x = Conv2D(
        filters=8*filters, kernel_size=(3, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    lvl5 = ELU()(x)

    x = MaxPooling2D(pool_size=(1, n_channels), padding='same')(lvl5)
    x = Conv2D(
        filters=4*filters, kernel_size=(3, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(
        filters=4*filters, kernel_size=(3, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(rate=0.5)(x)

    up4 = UpSampling2D(size=(4, 1))(x)
    att4 = AttentionPooling(
        filters=4*filters, channels=n_channels
    )([up4, lvl4])
    x = Concatenate(axis=-1)([up4, att4])
    x = Conv2D(
        filters=4*filters, kernel_size=(5, 1), strides=(1, 1),
        padding='same', activation=None
    )(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    up3 = UpSampling2D(size=(4, 1))(x)
    att3 = AttentionPooling(filters=4*filters, channels=n_channels)([up3, lvl3])
    x = Concatenate(axis=-1)([up3, att3])
    x = Conv2D(filters=4*filters, kernel_size=(7, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    up2 = UpSampling2D(size=(4, 1))(x)
    att2 = AttentionPooling(filters=4*filters, channels=n_channels)([up2, lvl2])
    x = Concatenate(axis=-1)([up2, att2])
    x = Conv2D(filters=4*filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    
    up1 = UpSampling2D(size=(4, 1))(x)
    att1 = AttentionPooling(filters=4*filters, channels=n_channels)([up1, lvl1])
    x = Concatenate(axis=-1)([up1, att1])
    x = Conv2D(filters=4*filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    up0 = UpSampling2D(size=(4, 1))(x)
    att0 = AttentionPooling(filters=4*filters, channels=n_channels)([up0, lvl0])
    x = Concatenate(axis=-1)([up0, att0])
    x = Conv2D(filters=4*filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(filters=4*filters, kernel_size=(15, 1), strides=(1, 1), padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(filters=1, kernel_size=(15, 1), strides=(1, 1), padding='same', activation='sigmoid')(x)

    network = Model(input_seq, x)

    return network