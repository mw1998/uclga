from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model


def MyCNN(input_shape=None, classes=4):
    img_input = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = Conv2D(32, (3, 3),
                      activation='relu',
                      padding='valid',
                      name='block1_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #7layer
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='valid',
                      name='block2_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #9layer
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = Conv2D(64, (3, 3),
                      activation='relu',
                      padding='valid',
                      name='block3_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #11layer
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Conv2D(128, (3, 3),
                      activation='relu',
                      padding='valid',
                      name='block4_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #13layer
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = Conv2D(256, (3, 3),
                      activation='relu',
                      padding='valid',
                      name='block5_conv2')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    #15layer
    # x = Conv2D(512, (3, 3),
    #                   activation='relu',
    #                   padding='same',
    #                   name='block6_conv1')(x)
    # x = Conv2D(512, (3, 3),
    #                   activation='relu',
    #                   padding='same',
    #                   name='block6_conv2')(x)

    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)

    # #17layer
    # x = Conv2D(1024, (3, 3),
    #                   activation='relu',
    #                   padding='same',
    #                   name='block7_conv1')(x)
    # x = Conv2D(1024, (3, 3),
    #                   activation='relu',
    #                   padding='same',
    #                   name='block7_conv2')(x)

    # x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block7_pool')(x)    

    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)


    inputs = img_input

    model = Model(inputs, x, name='MyCNN')
    return model

if __name__ == '__main__':
    model = MyCNN(input_shape=(128, 128, 3))
    model.summary()
