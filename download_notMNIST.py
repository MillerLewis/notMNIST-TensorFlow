#Download and extract dataset
import os
from six.moves.urllib.request import urlretrieve
import time
import sys

url = 'https://commondatastorage.googleapis.com/books1000/'

last_percent_reported = None
start_time = None

LARGE_EXPECTED_BYTES = 247336696
SMALL_EXPECTED_BYTES = 8458043
#A progress bar that we will use in the following download_attempt function
# to display how far through the download we are
def download_progress(count, block_size, total_size):
    global last_percent_reported
    global start_time
    percent = int(count * block_size * 100 / total_size)
    if count == 0:
        sys.stdout.write("Download has begun!\n")
        sys.stdout.flush()
        start_time = time.time()

    time_taken = time.time() - start_time
    
    if last_percent_reported != percent and time_taken != 0:
        speed = int(count * block_size / (1024 * 1024 * time_taken))
        sys.stdout.write("...%d%%\n" % percent)
        sys.stdout.write("Time Taken: %d seconds\n" % time_taken)
        sys.stdout.write("Estimated Download speed: %d MB/s\n\n" % speed)
        #sys.stdout.write("Estimated time until completion: %d" % time_left)
        sys.stdout.flush()


    last_percent_reported = percent
        
        

#Uses urlretrieve to attempt to download the notMNIST dataset from the url
# will not attempt to redownload unless requested and will check that
# if the file is already present, that is in fact the file that we want
def download_attempt(filename, expected_bytes, overwrite = False):
    download_url = url + filename

    #Check if file is present
    if os.path.exists(filename) and not overwrite:
        print("File already present, skipping download.")

    elif not os.path.exists(filename) or overwrite:
        print("Attempting to download: ", filename)
        filename, _ = urlretrieve(download_url, filename, reporthook = download_progress)
        print("Download Complete!")
        fileInfo = os.stat(filename)
        if fileInfo.st_size == expected_bytes :
            print("File present and verified!", filename)
        else:
            raise Exception("Failed to verify ", filename)

    return filename
    
""" TESTING PURPOSES """
#train_filename = download_attempt('notMNIST_large.tar.gz', LARGE_EXPECTED_BYTES, overwrite = True)
#test_filename = download_attempt('notMNIST_small.tar.gz', SMALL_EXPECTED_BYTES, overwrite = True)
    
        
