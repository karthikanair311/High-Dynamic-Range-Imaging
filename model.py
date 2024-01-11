import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.python.keras import layers
import custom_layers
from tensorflow.python.keras.layers import Conv2D, ReLU, LeakyReLU, Concatenate, SeparableConv2D, Add
from tensorflow.python.keras.models import Model
from config import Configuration
cfg = Configuration()


class DRSB(tf.keras.Model):
  def __init__(self, *args, **kwargs):
      super(DRSB, self).__init__(*args, **kwargs)
      nFeat = cfg.noFeat

      # F-1
      self.conv1 = Conv2D(nFeat, kernel_size=3, padding='same')
      # F0
      self.conv2 = Conv2D(nFeat, kernel_size=3, padding='same')
      self.att11 = Conv2D(nFeat*2, kernel_size=3, padding='same')
      self.att12 = Conv2D(nFeat, kernel_size=3, padding='same')
      self.att31 = Conv2D(nFeat*2, kernel_size=3, padding='same')
      self.att32 = Conv2D(nFeat, kernel_size=3, padding='same')

      self.DP1 = custom_layers.DilationPyramid([3,2,1,1], 32)
      self.DP2 = custom_layers.DilationPyramid([3,2,1,1], 32)
      self.DP3 = custom_layers.DilationPyramid([3,2,1,1], 32)
      self.DP4 = custom_layers.DilationPyramid([3,2,1,1], 32)

      # feature fusion (GFF) 
      self.GFF_3x3 = Conv2D(nFeat, kernel_size = 3,padding='same')
      self.conv_up = SeparableConv2D(32, kernel_size = 3,padding='same')
      self.conv3 = Conv2D(3, kernel_size = 3,padding='same')
      

      self.relus = LeakyReLU()
      self.add = Add()
      self.concats = Concatenate(axis=-1)

  def call(self,inputs):
    b,h,w,c = inputs[0].shape

    F2_ = self.relus(self.conv1(inputs[0]))
    F1_ = self.relus(self.conv1(inputs[1]))
    F3_ = self.relus(self.conv1(inputs[2]))

    F1_i = self.concats([F1_, F2_])
    F1_A = self.relus(self.att11(F1_i))
    F1_A = self.att12(F1_A)
    F1_A = tf.keras.activations.sigmoid(F1_A)
    F1_ = F1_ * F1_A

    F3_i = self.concats([F3_, F2_])
    F3_A = self.relus(self.att31(F3_i))
    F3_A = self.att32(F3_A)
    F3_A = tf.keras.activations.sigmoid(F3_A)
    F3_ = F3_ * F3_A

    F_ = self.concats([F1_, F2_, F3_])

    F_0 = self.conv2(F_)
    F_1 = self.DP1(F_0)
    F_2 = self.DP2(F_1)
    F_3 = self.DP3(F_2)
    F_4 = self.DP4(F_3)
    FF = self.concats([F_1, F_2, F_3, F_4])

    FF = self.relus(FF)
    FdLF = self.GFF_3x3(FF)
    FdLF = self.relus(FdLF)
    FDF = self.add([FdLF,F2_])
    
    us = self.conv_up(FDF)
    x_out = self.relus(us)
    x_out = self.conv3(x_out)
    x_out = ReLU()(x_out)
    return x_out


def get_model(x,y,z):
  model = DRSB()
  model.build([x,y,z])
  return model

