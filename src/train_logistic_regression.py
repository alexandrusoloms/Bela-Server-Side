import os

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


from .Spectrogram import Spectrogram
from .train_model_utils import YieldItems


def make_path(path_name):
    return os.path.join(os.path.dirname("__file__"), "", ) + path_name


def load_data_item(file_path):
    spectrogram_data = Spectrogram(file_path=file_path).process()
    shape = (len(spectrogram_data) // 40, 40)
    spectrogram_data = np.array(spectrogram_data).reshape(shape)
    return spectrogram_data


data_paths = [
    make_path(path_name="/homes/amsolomes1/birdSongData/ff1010bird/wav/"),
    make_path(path_name="/homes/amsolomes1/birdSongData/warblrb10k/wav/"),
]

master_bird_dataset = pd.concat([
    pd.read_csv("/homes/amsolomes1/birdSongData/ff1010bird/ff1010bird_labels.csv"),
    pd.read_csv("/homes/amsolomes1/birdSongData/warblrb10k/warblrb10k_labels.csv")
])

master_bird_dataset["itemid"] = master_bird_dataset["itemid"].apply(func=lambda x: str(x))

path_yield = YieldItems.yield_from_path(data_paths=data_paths, batch_size=100)

MAX_SHAPE = 950

logistic_regression_clf = LogisticRegression()

test_batch = list()
test_label = list()
index = 0

for i in path_yield:
    train_batch = list()
    train_label = list()
    print("loading data...")
    for (train_path, bird_id) in i:
        spec = load_data_item(file_path=train_path)
        spec = np.mean(spec, axis=1)[:MAX_SHAPE]
        spec = np.append(1, spec)
        if spec.shape[0] > MAX_SHAPE:

            if index == 0:
                # it is the test data
                test_batch.append(spec)
                test_label.append(master_bird_dataset[master_bird_dataset["itemid"] == bird_id]["hasbird"].values[0])
            else:
                train_batch.append(spec)
                train_label.append(master_bird_dataset[master_bird_dataset["itemid"] == bird_id]["hasbird"].values[0])
    if index > 0:
        # Training
        print("training model...", index)
        logistic_regression_clf.fit(train_batch, train_label)
    index += 1

with open("logistic_regression_clf.pickle", "wb") as handle:
    pickle.dump(logistic_regression_clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
