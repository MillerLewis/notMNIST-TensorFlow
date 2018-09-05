#This file is all about merging and pruning the datasets into one

import normalize_data_basic
import extract_notMNIST
import download_notMNIST
import numpy as np
from six.moves import cPickle as pickle
import hashlib

image_size = 28
pixel_depth = 255.0

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows,img_size,img_size),dtype = np.float32)
        labels = np.ndarray(nb_rows,dtype = np.int32)
    else:
        dataset, labels = None, None
        
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size = 0):
    #This merges all the pickled files into one larger array again
    #The valid size will be used to create an array for validation, hyperparameter tuning
    #The function returns the validation dataset along with labels, and a training dataset along with labels
    
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    
    #Next we specify how many images we want of each class
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
    
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    
    end_l = vsize_per_class + tsize_per_class
    
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                #next we will shuffle it for randomization
                np.random.shuffle(letter_set)
                
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class,:,:]
                    valid_dataset[start_v:end_v,:,:] = valid_letter
                    valid_labels[start_v:end_v] = label
                    
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                    
                train_letter = letter_set[vsize_per_class:end_l,:,:]
                train_dataset[start_t:end_t,:,:] = train_letter
                train_labels[start_t:end_t] = label
                
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print("Unable to process data from",pickle_file,":",e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels


def prune_datasets(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
    #This is to ensure that there is no overlapping of any of the datasets
    #
    #To do this we will use the hashlib module

    #Note that this method doesnt take into account that multiple things can output the same hash
    train_hashes = [hashlib.sha1(x).digest() for x in train_dataset]
    valid_hashes = [hashlib.sha1(x).digest() for x in valid_dataset]
    test_hashes  = [hashlib.sha1(x).digest() for x in test_dataset]
    
    valid_in_train = np.in1d(valid_hashes,train_hashes)
    test_in_train = np.in1d(test_hashes, train_hashes)
    test_in_valid = np.in1d(test_hashes,valid_hashes)
    
    #We want to keep valid items if they're false in valid_in_train
    #i.e, we keep things that are in valid_hashes but not in train_hashes
    valid_keep = ~valid_in_train
    #We will keep items in test that aren't in train and also not in valid
    test_keep = ~(test_in_train | test_in_valid)
    
    #This method keeps the training set the largest
    
    cleaned_valid_dataset = valid_dataset[valid_keep]
    cleaned_valid_labels = valid_labels[valid_keep]
    cleaned_test_dataset = test_dataset[test_keep]
    cleaned_test_labels = test_labels[test_keep]
    
    print("valid -> train overlap: %d samples" % valid_in_train.sum())
    print("test  -> train overlap: %d samples" % test_in_train.sum())
    print("test  -> valid overlap: %d samples" % test_in_valid.sum())
    print()
    print("Full dataset tensor of cleaned train dataset: ", train_dataset.shape)
    print("Full dataset tensor of cleaned valid dataset: ", cleaned_valid_dataset.shape)
    print("Full dataset tensor of cleaned test dataset: ", cleaned_test_dataset.shape)

    return train_dataset, train_labels, cleaned_valid_dataset, cleaned_valid_labels, cleaned_test_dataset , cleaned_test_labels

    
