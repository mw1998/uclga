from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from nets.vgg16 import VGG16
from nets.attention_gate import attention_block

def Unet(input_shape=(512,512,3), num_classes=21):
    inputs = Input(input_shape)
    #-------------------------------#
    #   获得五个有效特征层
    #   feat1   512,512,64
    #   feat2   256,256,128
    #   feat3   128,128,256
    #   feat4   64,64,512
    #   feat5   32,32,512
    #-------------------------------#
    feat1, feat2, feat3, feat4, feat5 = VGG16(inputs) 
      
    channels = [64, 128, 256, 512]

    # 32, 32, 512 -> 64, 64, 512
    P5_up = UpSampling2D(size=(2, 2))(feat5)
    attention_4 = attention_block(P5_up, feat4, 256, 4)
    # 64, 64, 512 + 64, 64, 512 -> 64, 64, 1024
    P4 = Concatenate(axis=3)([attention_4, P5_up])
    # 64, 64, 1024 -> 64, 64, 512
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(P4)
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(P4)

    # 64, 64, 512 -> 128, 128, 512
    P4_up = UpSampling2D(size=(2, 2))(P4)
    attention_3 = attention_block(P4_up, feat3, 128, 3)
    # 128, 128, 256 + 128, 128, 512 -> 128, 128, 768
    P3 = Concatenate(axis=3)([attention_3, P4_up])
    # 128, 128, 768 -> 128, 128, 256
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(P3)
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(P3)

    # 128, 128, 256 -> 256, 256, 256
    P3_up = UpSampling2D(size=(2, 2))(P3)
    attention_2 = attention_block(P3_up, feat2, 64, 2)
    # 256, 256, 256 + 256, 256, 128 -> 256, 256, 384
    P2 = Concatenate(axis=3)([attention_2, P3_up])
    # 256, 256, 384 -> 256, 256, 128
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(P2)
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(P2)

    # 256, 256, 128 -> 512, 512, 128
    P2_up = UpSampling2D(size=(2, 2))(P2)
    attention_1 = attention_block(P2_up, feat1, 32, 1)
    # 512, 512, 128 + 512, 512, 64 -> 512, 512, 192
    P1 = Concatenate(axis=3)([attention_1, P2_up])
    # 512, 512, 192 -> 512, 512, 64
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(P1)
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer = RandomNormal(stddev=0.02))(P1)

    # 512, 512, 64 -> 512, 512, num_classes
    P1 = Conv2D(num_classes, 1, activation="softmax")(P1)

    model = Model(inputs=inputs, outputs=P1)
    return model

