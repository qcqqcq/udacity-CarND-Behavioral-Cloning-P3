###########
##  
## Pre-process data for use in training image recognition model 
## Loads diving log, finds corresponding image, save in batches
## as pickle files.  The batches are used to store data in managable
## sizes but also to load into memory for batch learning.
## 
##
## Originally developed for Udacity Self Driving Car Nanodegree
##
## Author: Brandon Quach
## https://www.linkedin.com/in/bquach
###########

import glob
import os
import matplotlib.image as mpimg
import cv2
import numpy as np
import pandas as pd
import pickle
import time
import pdb


# Define new #rows, #columns after resizing
new_num_rows = int(160/8)
new_num_cols = int(320/8)

# Define paths
base_data_dir = '../../data/'
processed_dir = os.path.join(base_data_dir,'processed_sim')


def preprocess_one_img(img):
    '''Operations to perform on one image'''

    # Resize to reduce # pixels
    return cv2.resize(img,(new_num_cols,new_num_rows))



def preprocess_batch_img(img_array):
    '''Operations to perform on a batch of images at a time'''

    # Normalize by dividing by 255
    img_array = img_array / 255.0

    return img_array


def select_rows(dlog):
    '''Decide which driver logs rows to keep'''

    # Remove all rows in which steering angle is 0.
    # If there are too many steering angless = 0, the prior probability
    # of the angles is near 0 so the training algorithm would
    # have a preference to set steering angles near zero for greatest
    # MSE accuracy.  This goes against our approach of recovery driving
    # where we want the vehicle to make non-zero steering angles to stay
    # on course.  This is also somewhat of a shortcut to the idea of
    # of only recording during recover phase and not recording when 
    # the vehicle is heading towards the lane edge. Since heading 
    # towards the edge if often accomplished with steering angle 0,
    # this way, I just keep recording all the time and don't have to
    # toggle.
    dlog = dlog[dlog['steering'] != 0]

    return dlog


def get_log_sets():
    '''
    Load entire driving log since its low memory
    Keep only relevant rows and columns
    Shuffle order so batches are random
    Split into test/val/train
    '''
    # Driver logs can come in sessions, each in a dir labeled as 
    # sim_YYYYMMDD and having driving_log.csv and a subdir IMG
    # which contains the images
    # Iterate through all sessions and concat all driver logs
    dlog = []
    for subfolder in glob.iglob(os.path.join(base_data_dir,'sim*')):

        # Get filepath of the log file
        log_file = os.path.join(subfolder,'driving_log.csv')

        # Read CSV as a Pandas dataframe, add column names
        subfolder_dlog = pd.read_csv(log_file,index_col=None,header=None
                           ,names=['center','left','right','steering'
                                   ,'throttle','brake','speed'])
        dlog.append(subfolder_dlog)

    # Concatenate driver logs from all sessions
    dlog = pd.concat(dlog)

    # Select which rows of driver log to keep.
    dlog = select_rows(dlog)

    # Shuffle row order by randomly sampling all rows
    dlog = dlog.sample(frac=1) 



    #### Split into datasets train/test/val ####
    num_test = int(len(dlog)/10) # Save 1/10th of data as test
    num_val = 2*num_test # Save same fraction as validation
    dlog_test = dlog[:num_test]
    dlog_val = dlog[num_test:num_val]
    dlog_train = dlog[num_val:]

    return {'test':dlog_test,'val':dlog_val,'train': dlog_train}


def preprocess(dataset,batch_size=100):
    '''Main code that does the preprocessing.  Gets driver logs, get 
    through each log and finds corresponding image and steering angle
    
    Arguments
        dataset: choose either 'train','val', or 'test'
        batch_size: Number of rows for one batch. This is not only size
            of the saved pickle file but also that size of the batch
            used for batch training.  So choose a size that fits both.
    '''

    # Get driver logs
    dlog_dict = get_log_sets()

    # Set up holders for batch data
    img_batch = []
    steering_batch = []
    logs_with_images = [] # logs to keep if after filtering for existing images

    # Iterate over rows in the driving log
    for ind,one_log in dlog_dict[dataset].iterrows():

        # Get image file name and read into memory.  If file not found
        # print an error
        img_name = one_log['center']
        try:
            img = mpimg.imread(img_name)
        except FileNotFoundError:
            print('Image not found %s'%img_name)
            continue

        # Preprocess image at the one image at a time level
        img = preprocess_one_img(img)

        # Add image and steering angle to batch
        img_batch.append(img)
        steering_batch.append(one_log['steering'])
        logs_with_images.append(one_log)

        # When batch grows to batch_size, pickle data.  
        # Also reset image and steering angle batch to empty lists
        if len(img_batch) >= batch_size:
            img_batch,steering_batch = pickle_data(img_batch
                                       ,steering_batch,dataset)


    # Write last batch, smaller than batch_size
    pickle_data(img_batch,steering_batch,dataset)


    # Create a new driver log that excludes logs where image was not found
    # This allows removing of data by simply deleting the JPG and also
    # takes away the need to keep searching for images that are known
    # not to exist.
    # Does not overwrite original, so need to manually overwrite
    logs_with_images = pd.DataFrame(logs_with_images)
    logs_with_images.to_csv(os.path.join(processed_dir,'driving_log.filtered.csv'),index=False)


def pickle_data(img_batch,steering_batch,dataset):
    '''Write batch image data with steering angles into a pickle file.
       Files are written at batches to prevent very large files on disk
       and also doubles for the batch size for batch training 
    '''

    # Perform any preprocessing at the batch level
    img_batch = preprocess_batch_img(np.array(img_batch))
    steering_batch = np.array(steering_batch)

    # Format data into a dictionary with image data and steering angle
    data = {'features':img_batch,'labels':steering_batch}

    # Create unique file name for the saved pickle file
    data_file_name = '%s_%i_%s.pkl'%(dataset,len(img_batch),time.strftime("%Y%m%d_%H%M%S"))

    # Add path to the file name
    data_file_name = os.path.join(processed_dir,data_file_name)

    # Pickle
    with open(data_file_name,'wb') as f:
        pickle.dump(data,f)

    # Retun empty lists representing newly reset image and steering angle
    # batches
    return [],[]



def data_iterator(dataset,one_cycle_only=False
                   ,include_sample_weights=False):
    '''
    Used by modules outside this one.  Iterates through all the data
    in pickle files, one pickle file (and thus batch) at a time.

    Arguments:
        dataset: choose 'train','val','test'
        one_cycle_only:  If True, stop iterator after one cycle through
            pickle files.  If False, loop through data indefinately
        include_sample_weights: If True, include sample weights in output
    '''

    # Data path expression used to find relvant pickle files
    data_exp = os.path.join(processed_dir,'%s*.pkl'%dataset)
    cycle_num = 0

    # Cycle through pickled batches
    while 1:
        cycle_start_time = time.time() #note start time
        for data_file in glob.iglob(data_exp):

            # Load actual batch data
            with open(data_file,'rb') as f:
                data = pickle.load(f)
                X_batch,y_batch = data['features'],data['labels']

            # Yield batch of image data and steering angle data
            # Label as X and y in generic modeling nomenclature
            if include_sample_weights:
                sample_weights = np.sqrt(abs(y_batch))
                yield X_batch,y_batch,sample_weights
            else:
                yield X_batch,y_batch

        # Print to screen stats about iteration speed
        print('%s cycle %i finish in %0.1f sec'%(dataset,cycle_num,time.time() - cycle_start_time))
        cycle_num += 1

        # if one_cycle_only, break out of the infinite while 1 loop
        if one_cycle_only: break
            
def preprocess_all():
    preprocess('val',batch_size=1000)
    preprocess('test',batch_size=1000)
    preprocess('train',batch_size=1000)


if __name__ == '__main__': 

    #a = next(data_generator('train')) #test data_generator
    #dlog_dict = get_log_sets() # test getting of logs
    preprocess_all()

