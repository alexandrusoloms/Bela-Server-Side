import os
from random import shuffle
from .__init__ import make_batches


class YieldItems(object):

    @classmethod
    def yield_from_path(cls, data_paths, batch_size, random_init=10, split_ratio=.3):
        """
        yields ``batch_size`` data items at a time


        :param data_paths: [str] -> list of strings defining the paths of the data
        :param batch_size: int -> size of individual batch
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
        yield test_batches
        # finally, we yield each batch
        for batch in train_batches:
            yield batch
