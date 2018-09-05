""" MAIN """
#This file will be the main running file.
#Upon running this, we will attempt to download, extract, normalize, merge
# and then train the data.

import download_notMNIST
import extract_notMNIST
import normalize_data_basic as ndb
import merge_prune_basic as merge_prune
import numpy as np
import matplotlib.pyplot as plt

image_size = 28
pixel_depth = 255.0

train_size = 200000
valid_size = 10000
test_size = 10000

if __name__ == "__main__":

    """ GETTING THE DATASET """
    train_filename = download_notMNIST.download_attempt('notMNIST_large.tar.gz', download_notMNIST.LARGE_EXPECTED_BYTES)
    print()
    test_filename = download_notMNIST.download_attempt('notMNIST_small.tar.gz', download_notMNIST.SMALL_EXPECTED_BYTES)
    print()

    """ EXTRACTING THE DATASET """
    train_folders = extract_notMNIST.extract_attempt(train_filename, force = False)
    print()
    test_folders = extract_notMNIST.extract_attempt(test_filename, force = False)
    print()

    """ LOADING AND PICKLING THE DATASETS AS NUMPY ARRAYS """
    train_datasets = ndb.maybe_pickle(train_folders,45000)
    print()
    test_datasets = ndb.maybe_pickle(test_folders,1800)
    print()

    """ MERGING THE DATASETS AND CREATING A LABEL ARRAY TO USE LATER """
    valid_dataset, valid_labels, train_dataset, train_labels = merge_prune.merge_datasets(
      train_datasets, train_size, valid_size)
    print()
    ##We won't use any validation set on the testing datasets
    _, _, test_dataset, test_labels = merge_prune.merge_datasets(test_datasets, test_size)
    print()

    """ CHECKING SHAPES OF DATASETS """
    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    """ SHUFFLING THE DATASETS (RETAINING RELATIVE LABEL ORDER """
    def randomize(dataset, labels):
        permutation = np.random.permutation(labels.shape[0])
        shuffled_dataset = dataset[permutation,:,:]
        shuffled_labels = labels[permutation]
        return shuffled_dataset, shuffled_labels

    print("\nRandomizing datasets...")
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    print("Finished randomizing datasets")

    
