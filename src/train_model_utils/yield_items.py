import os
from random import shuffle
import numpy as np
from .__init__ import make_batches

from .Spectrogram import Spectrogram


class YieldItems(object):

    @classmethod
    def yield_from_path(cls, data_paths, batch_size, master_bird_dataset, max_shape, random_init=10, split_ratio=.3):
        """
        yields ``batch_size`` data items at a time


        :param data_paths: [str] -> list of strings defining the paths of the data
        :param batch_size: int -> size of individual batch
        :param master_bird_dataset: -> pandas dataframe containing the labels of each instance
        :param max_shape: int -> number of dimensions each instance should have
        :param random_init: int -> random initializer seed
        :param split_ratio: float ->

        :return: yield_generator, yielding one batch of data items at a time.
                 It will return N batches of size ``batch_size``
        """
        files_path = list(data_path + file
                          for data_path in data_paths
                          for file in os.listdir(data_path))
        # each file name is the id of the item, this `files_path` becomes
        # a tuple(file_path, file_name) -> in order to have access to its label
        files_path = list(tuple([file_name, file_name.split("wav/")[1].split(".")[0]])
                          for file_name in files_path)
        # shuffling files_path, using a seed ``random_init`` for
        # reproducible results while testing
        shuffle(files_path, random=random_init)
        # next, we create N batches of length ``batch_size`` using ``make_batches``
        batches = make_batches(list_of_items=files_path, batch_number=batch_size)
        # splitting batches into train, test and yielding test first
        length_of_batches = len(batches)
        test_ratio = split_ratio * length_of_batches
        test_batches, train_batches = batches[:test_ratio], batches[test_ratio:]
        # yield test first
        yield cls.apply(batch=test_batches, master_bird_dataset=master_bird_dataset, max_shape=max_shape)
        # finally, we yield each batch
        for batch in train_batches:
            yield cls.apply(batch=batch, master_bird_dataset=master_bird_dataset, max_shape=max_shape)

    @staticmethod
    def apply(batch, master_bird_dataset, max_shape):
        """
        method to ``apply`` on a batch

        :param batch: a list of tuples (bird_file_path, bird_id)
        :param master_bird_dataset: a pandas data frame
        :param max_shape: maximum number of dimension each instance should have
        :return: [spectrogram data], [labels]
        """
        batch_data = list()
        batch_labels = list()
        # each `batch` iterable contains a path and a bird_id.
        # we loop over this, using the path to make a spectrogram using `Spectrogram`
        # and use the bird_id to read the correct label, found in `master_bird_dataset`
        # we then populate these into `batch_data` and `batch_labels`
        for (path, bird_id) in batch:
            # creating a spectrogram:
            spectrogram_data = Spectrogram(file_path=path).process()
            # by default `Spectrogram` makes filters of 40 Bands
            shape = (len(spectrogram_data) // 40, 40)
            spectrogram_data = np.array(spectrogram_data).reshape(shape)
            spectrogram_data = np.mean(spectrogram_data, axis=1)[:max_shape]
            spectrogram_data = np.append(1, spectrogram_data)
            if spectrogram_data.shape[0] > max_shape:
                # populating spectrogram_data
                batch_data.append(spectrogram_data)
                # populating the real label of the file
                batch_labels.append(master_bird_dataset[master_bird_dataset["itemid"] == bird_id]["hasbird"].values[0])
        yield batch_data, batch_labels