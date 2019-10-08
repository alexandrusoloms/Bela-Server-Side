"""
this script trains a keras model on the ff1010bird and warblrb10k train set.
`YieldItems` yields data, label pairs of a set size. It uses `yield`, returning a generator, making this
memory efficient.
"""
import pickle
import numpy as np
import pandas as pd

from load_keras_model import create_model
from train_model_utils import make_path, YieldItems

MAX_SHAPE = 1000

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
model = create_model(MAX_SHAPE=MAX_SHAPE)

test_batch = list()
test_label = list()
load_test = True

for epoch in range(20):
    path_yield = YieldItems.yield_pre_computed_bela_spectrogram_from_path(data_paths=data_paths, batch_size=150,
                                                                          master_bird_dataset=master_bird_dataset,
                                                                          max_shape=MAX_SHAPE, yield_test=load_test)
    batch_count = 0
    for i in path_yield:
        print("loading data...")

        is_train, batch, label = i
        if is_train:
            batch = np.array(batch)
            label = np.array(label)
            random_indices = np.random.randint(0, len(test_batch), 500)
            history = model.fit(batch, label, validation_data=(np.array(test_batch)[random_indices], np.array(test_label)[random_indices]))

            batch_count += 1
        else:
            test_batch.extend(batch)
            test_label.extend(label)

    load_test = False

    # saving model
    model.save('keras_clf_final_model-1000x80-3x1-epoch-{}.h5'.format(epoch))  # creates a HDF5 file


del model  # deletes the existing model
# Saving batch and label to check accuracy later
with open("keras_test_data_batch.pickle", "wb") as handle:
    pickle.dump(test_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("keras_test_data_label.pickle", "wb") as handle:
    pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
