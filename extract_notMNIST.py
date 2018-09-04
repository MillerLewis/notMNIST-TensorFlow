#This file will be for extracting notMNIST large and small
#The folder is compressed as a tar file, so we will need to use the
# tar file module to extract it.
#We'd like the images to be stored in 2 seperate folders, and then each
# class of image will be in their own file. Thankfully, upon extracting the
# folder, we can see that the images are already in their own file.
# As a result, this means we have the class labels attached to them.

import os
import tarfile
import download_notMNIST

num_classes = 10
#This function will attempt to extract the folders out.
#We don't really want to extract the folders again if they're already there
# so this is a check that we'll have to do
#We'll also throw an exception if it looks like we didn't extract enough
# of the folders.
def extract_attempt(filename, force = False):
    #splitext splits into root and ending, we want to get rid of .tar.gz,
    # i.e. the root of the root
    storing_folder = os.path.splitext(os.path.splitext(filename)[0])[0]

    if os.path.isdir(storing_folder) and not force:
        print("Appears %s is already present. Skipping extraction of %s" % (root, filename))
    else:
        print("Extraction %s. This could take a long while" % (filename))
        tar = tarfile.open(filename)
        tar.extractall(".")
        tar.close

        #The following creates a list of the folders that were just extracted
        #It adds a folder's relative path to the list if it actually exists
        # where we want them to be stores
        data_folders = [
            os.path.join(storing_folder, d) for d in sorted(os.listdir(storing_folder)) if os.path.isdir(os.path.join(actual_folder,d))]

        if len(data_folders) != num_classes:
            raise Exception("We expected %d folders but only got %d" % (num_classes, len(data_folders)))

        print(data_folders)
        return data_folders            





train_filename = download_notMNIST.download_attempt('notMNIST_large.tar.gz', download_notMNIST.LARGE_EXPECTED_BYTES)
test_filename = download_notMNIST.download_attempt('notMNIST_small.tar.gz', download_notMNIST.SMALL_EXPECTED_BYTES)


train_folders = extract_attempt(train_filename)
test_folders = extract_attempt(test_filename)
