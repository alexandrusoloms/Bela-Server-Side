import os

import keras
from keras.layers import Conv2D, Dropout, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Sequential


def create_model(checkpoint='', MAX_SHAPE=None):
    if checkpoint:
        return load_model(checkpoint)
    else:
        assert MAX_SHAPE != None
        model = Sequential()
        # augmentation generator
        # code from baseline : "augment:Rotation|augment:Shift(low=-1,high=1,axis=3)"
        # keras augmentation:
        # preprocessing_function
        # convolution layers
        model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(MAX_SHAPE, 80, 1), ))  # low: try different kernel_initializer
        model.add(BatchNormalization())  # explore order of Batchnorm and activation
        model.add(LeakyReLU(alpha=.001))
        model.add(MaxPooling2D(pool_size=(3, 3)))  # experiment with using smaller pooling along frequency axis
        model.add(Conv2D(16, (3, 3), padding='valid'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(16, (3, 1), padding='valid'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))
        model.add(MaxPooling2D(pool_size=(3, 1)))

        model.add(Conv2D(16, (3, 1), padding='valid'))  # drfault 0.01. Try 0.001 and 0.001
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))
        model.add(MaxPooling2D(pool_size=(3, 1)))

        # dense layers
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))
        model.add(Dropout(0.5))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=.001))  # leaky relu value is very small experiment with bigger ones
        model.add(Dropout(0.5))  # experiment with removing this dropout
        model.add(Dense(1, activation='sigmoid'))

        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])

        # # prepare callback
        # histories = my_callbacks.Histories()

        model.summary()
        return model
