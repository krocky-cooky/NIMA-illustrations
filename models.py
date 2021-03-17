import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
from efficientnet.keras import EfficientNetB7,EfficientNetB4


class WideResBlock(Model):
    def __init__(
        self,
        input_channels,
        output_channels
    ):
        super().__init__()
        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)
        self.conv1 = kl.Conv2D(
            output_channels,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            use_bias = False
        )
        self.bn2 = kl.BatchNormalization()
        self.av2 = kl.Activation(tf.nn.relu)
        self.dropout = kl.Dropout(rate = 0.2)
        self.conv2 = kl.Conv2D(
            output_channels,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            use_bias = False
        )
        self.sc = self._scblock(input_channels,output_channels)
        self.add = kl.Add()

    def _scblock(self,input_channels,output_channels):
        if input_channels == output_channels:
            return lambda x: x
        else:
            conv_sc = kl.Conv2D(
                output_channels,
                kernel_size = 3,
                strides = 1,
                padding = 'same',
                use_bias = False
            )
            return conv_sc

    def call(self,x):
        out1 = self.conv1(self.av1(self.bn1(x)))
        out2 = self.conv2(self.dropout(self.av2(self.bn2(out1)),training = True))
        out = self.add([out2,self.sc(x)])
        return out

class ResBlock(Model):
    def __init__(
        self,
        input_channels,
        output_channels,
    ):
        super().__init__()
        bneck_channels = output_channels // 4
        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)
        self.conv1 = kl.Conv2D(
            bneck_channels,
            kernel_size = 1,
            strides = 1,
            padding = 'valid',
            use_bias = False
        )
        self.bn2 = kl.BatchNormalization()
        self.av2 = kl.Activation(tf.nn.relu)
        self.conv2 = kl.Conv2D(
            bneck_channels,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            use_bias = False
        )
        self.bn3 = kl.BatchNormalization()
        self.av3 = kl.Activation(tf.nn.relu)
        self.conv3 = kl.Conv2D(
            output_channels,
            kernel_size = 1,
            strides = 1,
            padding = 'valid',
            use_bias = False
        )

        self.shortcut = self._scblock(input_channels,output_channels)
        self.add = kl.Add()

       

    def _scblock(
        self,
        input_channels,
        output_channels
    ):
        if input_channels == output_channels:
            return lambda x : x
        else:
            self.bn_sc = kl.BatchNormalization()
            self.conv_sc = kl.Conv2D(
                output_channels,
                kernel_size = 1,
                strides = 1,
                padding = 'same',
                use_bias = False
            )
            return self.conv_sc

    def call(self,x):
        out1 = self.conv1(self.av1(self.bn1(x)))
        out2 = self.conv2(self.av2(self.bn2(out1)))
        out3 = self.conv3(self.av3(self.bn3(out2)))
        shortcut = self.shortcut(x)
        out4 = self.add([out3, shortcut])
        
        return out4

class ResNet(Model):
    def __init__(
        self,
        input_shape,
        output_dim,
    ):
        super().__init__()
        self._layers = [
            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(
                filters = 16,
                kernel_size = 7,
                strides = 2,
                padding = 'same',
                input_shape = input_shape
            ),
            ResBlock(16,32),
            [
                ResBlock(32,32) for _ in range(4)
            ],
            kl.Conv2D(
                64,
                kernel_size = 1,
                strides = 2,
                use_bias = False
            ),
            [
                ResBlock(64,64) for _ in range(2)
            ],
            kl.Conv2D(
                128,
                kernel_size = 1,
                strides = 2,
                use_bias = False
            ),
            [
                ResBlock(128,128) for _ in range(3)
            ],
            kl.GlobalAveragePooling2D(),
            kl.Dense(1000,activation="relu"),
            kl.Dense(output_dim,activation = 'softmax')
        ]

    def call(self,x):
        for layer in self._layers:
            if isinstance(layer,list):
                for l in layer:
                    x = l(x)
            else:
                x = layer(x)
        return x

class WideResNet(Model):
    def __init__(
        self,
        input_shape,
        output_dim,
    ):
        super().__init__()
        self._layers = [
            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(
                filters = 16,
                kernel_size = 3,
                strides = 1,
                padding = 'same',
                input_shape = input_shape,
                use_bias = False
            ),
            kl.MaxPool2D(pool_size=3, strides=2, padding="same"),
            WideResBlock(16,32),
            [
                WideResBlock(32,32) for _ in range(1)
            ],
            kl.Conv2D(
                64,
                kernel_size = 1,
                strides = 2,
                use_bias = False
            ),
            [
                WideResBlock(64,64) for _ in range(2)
            ],
            kl.Conv2D(
                128,
                kernel_size = 1,
                strides = 2,
                use_bias = False
            ),
            [
                WideResBlock(128,128) for _ in range(2)
            ],
            kl.GlobalAveragePooling2D(),
            kl.Dense(1000,activation = 'relu'),
            kl.Dense(output_dim,activation = 'softmax')
        ]
        
    def call(self,x):
        for layer in self._layers:
            if isinstance(layer,list):
                for l in layer:
                    x = l(x)
            else:
                x = layer(x)
        return x

class WideResNetWithMultiOutput(Model):
    def __init__(
        self,
        input_shape,
        output_dim,
    ):
        super().__init__()
        self._layers = [
            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(
                filters = 16,
                kernel_size = 3,
                strides = 1,
                padding = 'same',
                input_shape = input_shape,
                use_bias = False
            ),
            kl.MaxPool2D(pool_size=3, strides=2, padding="same"),
            WideResBlock(16,32),
            [
                WideResBlock(32,32) for _ in range(1)
            ],
            kl.Conv2D(
                64,
                kernel_size = 1,
                strides = 2,
                use_bias = False
            ),
            [
                WideResBlock(64,64) for _ in range(2)
            ],
            kl.Conv2D(
                128,
                kernel_size = 1,
                strides = 2,
                use_bias = False
            ),
            [
                WideResBlock(128,128) for _ in range(2)
            ],
            kl.GlobalAveragePooling2D()
        ]
        self.bookmark_encode = kl.Dense(1000,activation = 'relu')
        self.bookmark_output = kl.Dense(output_dim,activation = 'softmax',name = 'bookmark')
        self.aspect_ratio_encode = kl.Dense(1000,activation = 'relu')
        self.aspect_ratio_output = kl.Dense(1,activation = 'linear',name = 'aspect_ratio')
        
    def call(self,x):
        for layer in self._layers:
            if isinstance(layer,list):
                for l in layer:
                    x = l(x)
            elif isinstance(layer,dict):
                print(layer)
                raise Exception('out')
            else:
                x = layer(x)
        bookmark = self.bookmark_output(self.bookmark_encode(x))
        aspect_ratio = self.aspect_ratio_output(self.aspect_ratio_encode(x))

        return [bookmark,aspect_ratio]



def EfficientNet(
    input_shape,
    output_dim
):
    input_layer = kl.Input(shape = input_shape)
    efficient_net = EfficientNetB4(
        weights = 'imagenet',
        include_top = False,
        input_tensor = input_layer,
        pooling = 'max'
    )

    for layer in efficient_net.layers:
        layer.trainable = True

    bottleneck = efficient_net.output

    _ = kl.Dense(1000,activation = 'relu')(bottleneck)
    bookmark_output = kl.Dense(output_dim,activation = 'softmax',name = 'bookmark')(_)

    #_ = kl.Dense(1000,activvation = 'relu')(bottleneck)
    #aspect_ratio_output = kl.Dense(2,activation = 'softmax',name = 'aspect_ratio')(_)

    model = Model(inputs = input_layer,outputs = bookmark_output)

    return model


def EfficientNetWithMultiOutput(
    input_shape,
    output_dim,
):
    input_layer = kl.Input(shape = input_shape)
    efficient_net = EfficientNetB7(
        weights = 'imagenet',
        include_top = False,
        input_tensor = input_layer,
        pooling = 'max'
    )

    for layer in efficient_net.layers:
        layer.trainable = True

    bottleneck = efficient_net.output

    _ = kl.Dense(1000,activation = 'relu')(bottleneck)
    bookmark_output = kl.Dense(output_dim,activation = 'softmax',name = 'bookmark')(_)

    _ = kl.Dense(1000,activvation = 'relu')(bottleneck)
    aspect_ratio_output = kl.Dense(2,activation = 'softmax',name = 'aspect_ratio')(_)

    model = Model(inputs = input_layer,outputs = [bookmark_output,aspect_ratio_output])

    return model



def EMD(t,preds):
    return tf.reduce_mean(tf.reduce_sum(tf.math.cumsum(preds-t,axis = 1)**2,axis = 1))

    
