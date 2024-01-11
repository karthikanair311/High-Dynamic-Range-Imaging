import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Conv2D, LeakyReLU, Concatenate, Add, SeparableConv2D
import tensorflow_addons as tfa


class DilationPyramid(tf.keras.layers.Layer):
  def __init__(self, dilation_rates = [3,2,1,1], n_filters=32):
    super(DilationPyramid, self).__init__()
    self.convs = [SeparableConv2D(n_filters,3,1,'same',dilation_rate = i) for i in dilation_rates]
    self.relus = [LeakyReLU() for i in range(len(dilation_rates))]
    self.concats = [Concatenate(axis=-1) for i in range(len(dilation_rates))]
    self.add = Add()
  
  def build(self, input_shape):
    self.bottleneck = Conv2D(input_shape[-1],1,1,'same')

  def call(self,x_in):
    x_in_input = x_in
    in_channels = x_in.shape[-1]
    for i in range(len(self.convs)):
        x_out = self.relus[i](x_in)
        x_out = self.convs[i](x_out)
        x_in = self.concats[i]([x_in,x_out])
    x_out = self.bottleneck(x_in) # bottleneck
    x_out = self.add([x_in_input,x_out])
    return x_out

