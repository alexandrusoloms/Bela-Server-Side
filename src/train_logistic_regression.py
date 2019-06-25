import os

import numpy as np
import pandas as pd
from random import shuffle
from matplotlib import pyplot as plt

from Spectrogram import Spectrogram

from sklearn.linear_model import LogisticRegression
import pickle


def make_path(path_name):
    return os.path.join(os.path.dirname("__file__"), "", ) + path_name


def make_batches(list_of_items, number_of_batches):
    """
    separate a list of items into `number_of_batches parts

    :param list_of_items: [
                            object1,
                            object2,
                            object3,
                            ...
                          ]
    :param number_of_batches: <int> / <floats>
    :return:
    """
    return [list_of_items[i:i + number_of_batches] for i in range(0, len(list_of_items), number_of_batches)]



class YieldPaths(object):
    
    @classmethod
    def yield_paths(cls, data_paths, batch_num=10):
        
        files_full_path = list(data_path + file  
                           for data_path in data_paths 
                           for file in os.listdir(data_path))
        
        files_full_path = list(tuple([file_name, file_name.split("txt/")[1].split(".")[0]])
                              for file_name in files_full_path)
        shuffle(files_full_path)
        batches  = make_batches(list_of_items=files_full_path,
                               number_of_batches=batch_num)
        
        for (batch) in batches:
            yield batch
            

            
def load_data_item(file_path):
    
    with open(file_path, "r") as handle:
        spec = handle.read().split()
        
    spec = [eval(x) for x in spec]
    shape = (len(spec) // 40, 40)
    spec = np.array(spec).reshape(shape)
    return spec


data_paths = [
    make_path(path_name="/homes/amsolomes1/birdSongData/ff1010bird/txt/"),
    make_path(path_name="/homes/amsolomes1/birdSongData/warblrb10k/txt/"),
#     make_path(path_name="/homes/amsolomes1/birdSongData/birdVox-DCASE/txt/")
]

master_bird_dataset = pd.concat([
    pd.read_csv("/homes/amsolomes1/birdSongData/ff1010bird/ff1010bird_labels.csv"),
    pd.read_csv("/homes/amsolomes1/birdSongData/warblrb10k/warblrb10k_labels.csv")
])
master_bird_dataset["itemid"] = master_bird_dataset["itemid"].apply(func=lambda x: str(x))

path_yielder = YieldPaths.yield_paths(data_paths=data_paths, batch_num=100)


MAX_SHAPE = 950

logistic_regression_clf = LogisticRegression()
index = 0

for i in path_yielder:
    
    train_batch = list()
    train_label = list()
    
    print("loading data...")
    for (train_path, bird_id) in i:
        spec = load_data_item(file_path=train_path)
        spec = np.mean(spec, axis=1)[:MAX_SHAPE]
        spec = np.append(1, spec)
        if spec.shape[0] > MAX_SHAPE:
            train_batch.append(spec)
            train_label.append(master_bird_dataset[master_bird_dataset["itemid"] == bird_id]["hasbird"].values[0])
    
    # Training
    print("training model...")
    logistic_regression_clf.fit(train_batch, train_label)
    
    

with open("logistic_regression_clf.pickle", "wb") as handle:
    pickle.dump(logistic_regression_clf, handle, protocol=pickle.HIGHEST_PROTOCOL)