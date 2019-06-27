import pickle
import numpy as np
import pandas as pd

import keras
from keras.layers import Dense, GRU, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU

from train_model_utils import make_path, YieldItems


MAX_SHAPE = 950

data_paths = [
    make_path(path_name="/homes/amsolomes1/birdSongData/ff1010bird/txt/"),
    make_path(path_name="/homes/amsolomes1/birdSongData/warblrb10k/txt/"),
]

master_bird_dataset = pd.concat([
    pd.read_csv("/homes/amsolomes1/birdSongData/ff1010bird/ff1010bird_labels.csv"),
    pd.read_csv("/homes/amsolomes1/birdSongData/warblrb10k/warblrb10k_labels.csv")
])

master_bird_dataset["itemid"] = master_bird_dataset["itemid"].apply(func=lambda x: str(x))

# `YieldItems` will implement the feature extraction process from a wav file
#  it will create N batches of length `batch_size` containing:
#           1. numerical representation of a spectrogram of shape (MAX_SHAPE+1, 1)
#           2. a label
#
path_yield = YieldItems.yield_pre_computed_bela_spectrogram_from_path(data_paths=data_paths, batch_size=100,
                                                                      master_bird_dataset=master_bird_dataset,
                                                                      max_shape=MAX_SHAPE)

# MODEL
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(MAX_SHAPE, 40)))
model.add(Dropout(0.2))
model.add(GRU(100, return_sequences=False))
model.add(Dropout(0.2))

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

model.summary()

test_batch = list()
test_label = list()

# The first item yielded from `YieldItems` is the test batch and test label,
# and so we need to take it into account
for i in path_yield:
    print("loading data...")

    is_train, batch, label = i
    if is_train:
        batch = np.array(batch)
        label = np.array(label)
        model.fit(batch, label)
    else:
        test_batch.extend(batch)
        test_label.extend(label)

# saving model
model.save('keras_gru_clf.h5')  # creates a HDF5 file
del model  # deletes the existing model
