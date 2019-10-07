"""
this script trains a keras model on the ff1010bird and warblrb10k train set.
`YieldItems` yields data, label pairs of a set size. It uses `yield`, returning a generator, making this
memory efficient.
"""
import pickle
import numpy as np
import pandas as pd

import keras
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, AveragePooling2D
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

from train_model_utils import make_path, YieldItems

MAX_SHAPE = 998

data_paths = [
    make_path(path_name="/homes/amsolomes1/birdSongData/ff1010bird/txt_2/"),
    make_path(path_name="/homes/amsolomes1/birdSongData/warblrb10k/txt_2/"),
    make_path(path_name="/homes/amsolomes1/birdSongData/birdVox-DCASE/txt_2/"),
]

master_bird_dataset = pd.concat([
    pd.read_csv("/homes/amsolomes1/birdSongData/ff1010bird/ff1010bird_labels.csv"),
    pd.read_csv("/homes/amsolomes1/birdSongData/warblrb10k/warblrb10k_labels.csv"),
    pd.read_csv("/homes/amsolomes1/birdSongData/birdVox-DCASE/BirdVox-DCASE-20k.csv")
])

master_bird_dataset["itemid"] = master_bird_dataset["itemid"].apply(func=lambda x: str(x))

# `YieldItems` will implement the feature extraction process from a wav file
#  it will create N batches of length `batch_size` containing:
#           1. numerical representation of a spectrogram of shape (MAX_SHAPE+1, 1)
#           2. a label
#

# KERAS MODEL

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

test_batch = list()
test_label = list()
load_test = True

for epoch in range(20):
    path_yield = YieldItems.yield_pre_computed_bela_spectrogram_from_path(data_paths=data_paths, batch_size=150,
                                                                          master_bird_dataset=master_bird_dataset,
                                                                          max_shape=MAX_SHAPE, yield_test=load_test)
    # test_batch = list()
    # test_label = list()

    # The first item yielded from `YieldItems` is the test batch and test label,
    # and so we need to take it into account
    batch_count = 0
    for i in path_yield:
        print("loading data...")

        is_train, batch, label = i
        if is_train:
            batch = np.array(batch)
            label = np.array(label)
            random_indices = np.random.randint(0, len(test_batch), 500)
            history = model.fit(batch, label, validation_data=(np.array(test_batch)[random_indices], np.array(test_label)[random_indices]))

            # with open("keras_history-batch-{}-epoch-{}.pickle".format(batch_count, epoch), "wb") as handle:
            #     pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

            batch_count += 1
        else:
            test_batch.extend(batch)
            test_label.extend(label)

    load_test = False

    # saving model
    model.save('keras_clf_final_model-epoch-{}.h5'.format(epoch))  # creates a HDF5 file


del model  # deletes the existing model
# Saving batch and label to check accuracy later
with open("keras_test_data_batch.pickle", "wb") as handle:
    pickle.dump(test_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("keras_test_data_label.pickle", "wb") as handle:
    pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
