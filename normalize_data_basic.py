#Normalize data basic
from six.moves import cPickle as pickle
import matplotlib.image as mpimg
import download_notMNIST
import extract_notMNIST
import os
import numpy as np

image_size = 28
pixel_depth = 255.0
def load_letter(folder, min_num_images):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape = (len(image_files), image_size, image_size), dtype = np.float32)
    
    print(folder)
    num_images = 0
    
    for image in image_files:
        image_file = os.path.join(folder,image)
        try:
            if num_images % 1000 == 0:
                print (num_images)
            image_data = mpimg.imread(image_file)-pixel_depth/2/pixel_depth
            
            if image_data.shape != (image_size, image_size):
                raise Exception("Unexpectedf image shape: %s" % str(image_file.shape))
        
            dataset[num_images,:,:] = image_data
            num_images += 1
        
        except (IOError, ValueError) as e:
            print("Could not read: ", image_file, ":", e, "-it's okay, skipping.")
            
    dataset = dataset[0:num_images,:,:]
    if num_images < min_num_images:
        raise Exception("Many fewer images than expected: %d < %d" % num_images, min_num_images)
        
    print("Full data set tensor: ", dataset.shape)
    print("Mean: ", np.mean(dataset))
    print("Standard deviation: ", np.std(dataset))
    return dataset

def maybe_pickle(data_folders,min_num_images_per_class, force = False):
    dataset_names = []
    
    for folder in data_folders:
        set_filename = folder + ".pickle"
        dataset_names.append(set_filename)
        
        if os.path.exists(set_filename) and not force:
            print("%s already present - skipping pickling." % set_filename)
        
        else:
            print("Pickling %s." % set_filename)
            dataset = load_letter(folder,min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print("Unable to save data to: ", set_filename, " : ", e)

    return dataset_names

"""
train_filename = download_notMNIST.download_attempt('notMNIST_large.tar.gz', download_notMNIST.LARGE_EXPECTED_BYTES)
test_filename = download_notMNIST.download_attempt('notMNIST_small.tar.gz', download_notMNIST.SMALL_EXPECTED_BYTES)

train_folders = extract_notMNIST.extract_attempt(train_filename, force = False)
test_folders = extract_notMNIST.extract_attempt(test_filename, force = False)

train_datasets = maybe_pickle(train_folders,45000)
test_datasets = maybe_pickle(test_folders,1800)"""
