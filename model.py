###########
##  
## Main code to define model, train model, and make predictions 
## using model
##
## Originally developed for Udacity Self Driving Car Nanodegree
##
## Author: Brandon Quach
## https://www.linkedin.com/in/bquach
###########

# Import custom modules
import PreProcess as pp

# Import standard helper modules
import numpy as np
import pandas as pd
import pickle
import glob
import time

# Import keras modules
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.layers.pooling import MaxPooling2D



# Define the convulutional model
def define_conv_model(input_rows,input_cols):
    '''Define convolutional model based on ResNet. Use this space to 
    add layers, max pooling, dropout, etc.  Returns the model.  

    Arguments:
        input_rows: Number of rows in image AFTER post-processing
        input_cols: Number of columns in image AFTER post-processing
    '''


    # Choose sequantial model
    model = Sequential()



    ##### Convlutional layers #####
    # Add (5 x 5) x 6 conv layer with ReLu
    # Output (16 x 36) x 6 maps
    model.add(Convolution2D(6,5,5,input_shape=(input_rows,input_cols,3),init='normal'))
    model.add(Activation('relu'))

    # Pool down to (8 x 18) x 6 maps
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    # Add (5 x 5) x 16 conv layer with ReLu
    # Output (4 x 14) x 16 maps
    model.add(Convolution2D(16,5,5,init='normal'))
    model.add(Activation('relu'))

    # Pool down to (2 x 7) x 16 maps
    model.add(MaxPooling2D(pool_size=(2,2),))
    model.add(Dropout(0.2))




    ######## Fully connected layers #####

    # Flatten to 224 nodes
    model.add(Flatten())

    # FC layer with 50 nodes and ReLu
    model.add(Dense(50,init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

   
    # FC layer with 15 nodes and ReLu
    model.add(Dense(15,init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))




    ######## Output model #########

    model.add(Dense(1,init='normal'))
    # Tried softmax before but predictions would keep at 1 all the time
    #model.add(Activation('softmax')) #decided against softmax
    
    model.summary() #print output to screen for reference
    
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(),
                  metrics=['mean_absolute_error'])
    return model



def train_model(test,save=False):
    '''Train model.  
     
    Arguments:
        test:  For debugging.  If True, use minimal # epochs to quickly 
            go through code to check for crashes, etc.
        save:  If True, save model and history
    '''

    # Test mode to test the pipline for crashes, etc
    if test:
        epochs = 2
        samples_per_epoch = 2
    else:
        # Real mode
        epochs = 60 # put a large number here since early stopping is used
        samples_per_epoch = 10000 # sort of roughly equal to number of training samples, but does not have to be that exact


    # Define model. See define_conv_model for details
    model = define_conv_model(pp.new_num_rows,pp.new_num_cols)

    # Early stopping when validation loss does not decrease
    early_stopping = EarlyStopping(monitor='val_loss', patience=4,verbose=2)

    # Data generators defined in the PreProcess module output batches
    # of data at a time instead of loading all data into memory
    train_generator = pp.data_iterator('train',include_sample_weights=True)
    validation_generator = pp.data_iterator('val',include_sample_weights=True)

    # The actual training/fitting code
    history = model.fit_generator(train_generator
            ,samples_per_epoch=samples_per_epoch
            ,nb_epoch=epochs,verbose=2
            ,validation_data = validation_generator
            ,nb_val_samples = 1600
            ,callbacks=[early_stopping])


    print('Model building done')
    
    # Save model
    if save:
        '''
        # Save model structure to JSON
        model_json = model.to_json()
        with open('model.json','w') as json_file:
            json_file.write(model_json)

        # Save weights to HDF5
        model_weights = 'model.h5'
        model.save_weights(model_weights)
        print('model saved')
        '''
        # Save model to HDF5
        model_file_name = 'model.h5'
        model.save(model_file_name)
        print('model saved')

        # Save history as pickle
        # Saving the model in the history can take a lot of space
        # Since model is saked elsewhere, we overwrite it with None
        history.model = None
        with open('history.pkl','wb') as f:
            pickle.dump(history,f)
        print('history saved')
        
    else:
        print('Save set to False so not saving model')

    return model,history


def make_predictions():
    '''Score all data and save to CSV for analysis'''

    all_predictions = [] # contains predictions for all 3 datasets
    for dataset in ['test','val','train']:

        # Initialize real values and predicted values
        y = np.array([]) # the real value 
        y_hat = np.array([]) # the predicted value

        # Iterate over the data in batches using the data generator
        for X_batch,y_batch in pp.data_iterator(dataset,one_cycle_only=True):
            y_hat_batch = model.predict(X_batch).flatten()
            y = np.append(y,y_batch)
            y_hat = np.append(y_hat,y_hat_batch)
    
        # Fomat into Pandas dataframe
        predictions = pd.DataFrame(np.vstack([y,y_hat]).T)
        predictions.columns = ['y','y_hat']
        predictions['dataset'] = dataset
        all_predictions.append(predictions)

    # Concat all predictions together and save to CSV
    all_predictions = pd.concat(all_predictions)
    all_predictions.to_csv('predictions.csv')


if __name__ == '__main__':

    test = False
    model,history = train_model(test,save=True)

    print('')
    print('')
    print('Start Evaluation')
    make_predictions()
