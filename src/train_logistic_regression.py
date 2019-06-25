"""
this script trains a logistic regression on the ff1010bird and warblrb10k train set.

`YieldItems` yields data, label pairs of a set size. It uses `yield`, returning a generator, making this
memory efficient.
"""
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


from train_model_utils import make_path, YieldItems


MAX_SHAPE = 950

data_paths = [
    make_path(path_name="/homes/amsolomes1/birdSongData/ff1010bird/wav/"),
    make_path(path_name="/homes/amsolomes1/birdSongData/warblrb10k/wav/"),
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
path_yield = YieldItems.yield_from_path(data_paths=data_paths, batch_size=100, master_bird_dataset=master_bird_dataset,
                                        max_shape=MAX_SHAPE)

test_batch = list()
test_label = list()
train_batch = list()
train_label = list()

index = 0
logistic_regression_clf = LogisticRegression()
# The first item yielded from `YieldItems` is the test batch and test label,
# and so we need to take it into account
for i in path_yield:
    print("loading data...")
    if index != 0:
        train_batch, train_label = i
        logistic_regression_clf.fit(train_batch, train_label)
    else:
        test_batch, test_label = i
    index += 1

# saving model
with open("logistic_regression_clf.pickle", "wb") as handle:
    pickle.dump(logistic_regression_clf, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Saving batch and label to check accuracy later
with open("test_data_batch.pickle", "wb") as handle:
    pickle.dump(test_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("test_data_label.pickle", "wb") as handle:
    pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
