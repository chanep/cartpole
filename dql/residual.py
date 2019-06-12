import keras
from keras.layers import Activation
from keras.layers import Conv2D, Add, BatchNormalization
from keras.engine.topology import Layer


# Define the residual block as a new layer
class Residual(Layer):
    def __init__(self, channels_in,kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

    def call(self, x):
        # the residual block using Keras functional API
        first_layer = Activation("linear", trainable=False)(x)
        x = Conv2D( self.channels_in,
                    self.kernel,
                    padding="same")(first_layer)
        # x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D( self.channels_in,
                    self.kernel,
                    padding="same")(x)
        # x = BatchNormalization()(x)
        residual = Add()([x, first_layer])
        x = Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
