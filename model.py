import argparse
import csv
import cv2
import h5py
from keras.layers import Activation, Cropping2D, Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import __version__ as keras_version
import numpy as np
import os
from random import randint
import sklearn

def batch_gen(log_samples, batch_size=32, shuffle=False, horizontal_flip=False):
    
    offcenter_camera_angle_delta = 0.1
    
    num_samples = len(log_samples)
    #batch_X = np.empty((batch_size*3, 160, 320, 3))
    #batch_y = np.empty((batch_size*3))
    
    # epoch loop
    while 1:
        if shuffle:
            sklearn.utils.shuffle(log_samples)
        
        # batch loop
        for offset in range(0, num_samples, batch_size):
            batch_X = []
            batch_y = []
            
            end = offset + batch_size
            batch_samples = log_samples[offset:end]
            
            # sample pre-processing loop
            # NOTE: preprocessing done here will NOT be included during prediction (i.e. actual driving)
            for sample_idx, batch_sample in enumerate(batch_samples):
                # every sample has 3 images
                idx = 3 * sample_idx
                
                # load images from file
                center_image = load_image(batch_sample[0])
                left_image = load_image(batch_sample[1])
                right_image = load_image(batch_sample[2])
                
                # assign to batch
                batch_X.append(center_image)
                batch_X.append(left_image)
                batch_X.append(right_image)
                    
                # extract label
                steering_angle = float(batch_sample[3])
                batch_y.append(steering_angle)
                batch_y.append(steering_angle + offcenter_camera_angle_delta + 0.3*steering_angle**2)
                batch_y.append(steering_angle - offcenter_camera_angle_delta - 0.3*steering_angle**2)
                    
            if shuffle:
                batch_X, batch_y = sklearn.utils.shuffle(batch_X, batch_y)
                
            batch_X = np.asarray(batch_X)
            batch_y = np.asarray(batch_y)
            
            # 50% chance to flip images (reduce directional bias)
            if horizontal_flip:
                for idx in range(len(batch_X)):
                    if randint(0, 1) == 0:
                        batch_X[idx] = np.fliplr(batch_X[idx])
                        batch_y[idx] = -batch_y[idx]
            
            yield batch_X, batch_y

def load_image(path):
    if not os.path.exists(path):
        print("ERROR - image path not found: ", path)
        return None
    bgr_image = cv2.imread(path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def load_h5_model(filename):
    # check that model Keras version is same as local Keras version
    f = h5py.File(filename, mode='r')
    model_keras_version = f.attrs.get('keras_version')
    imported_keras_version = str(keras_version).encode('utf8')

    if model_keras_version != imported_keras_version:
        print('You are using Keras version ', imported_keras_version,
              ', but the model was built using ', model_keras_version)

    live_model = load_model(filename)
    return live_model

def save_h5_model(model, filename):
    print()
    print("saving model to ./{0}...".format(filename))
    model.save(filename)
    print("model saved")
    return

def read_driving_log(csvfile, skip_header=True):
    samples = []
    with open(csvfile) as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader)
        for line in reader:
            samples.append(line)
    return samples

def new_model_architecture():
    model = Sequential()
    
    # crop and normalize
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    
    # convolutional layers
    model.add(Convolution2D(nb_filter=10, nb_row=5, nb_col=5, subsample=(2,2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=20, nb_row=3, nb_col=3, subsample=(2,2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    
    # dense layers
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    
    # predicting one value
    model.add(Dense(1))
    
    return model

def main(args):
    # load model or create new one
    if os.path.isfile(args.model):
        print("continuing training for {0}".format(args.model))
        model = load_h5_model(args.model)
    else:
        print("training new model")
        model = new_model_architecture()
    
    # multiple logs, each in a subfolder of args.log_folder
    logs = os.listdir(args.log_folder)
    for i, log in enumerate(logs):
        logs[i] = os.path.join(args.log_folder, log)
    
    # load and process log files
    valid_fraction=0.25
    train_samples = []
    valid_samples = []
    for log_dir in logs:
        # load a log file
        log_path = os.path.join(log_dir, 'driving_log.csv')
        log = read_driving_log(log_path)
        
        # calculate validation data indicies
        n_samples = len(log)
        n_valid_samples = int(n_samples*valid_fraction)
        valid_start_idx = randint(0, n_samples-n_valid_samples-1)
        valid_end_idx = valid_start_idx + n_valid_samples
        
        img_folder = os.path.join(log_dir, 'IMG')
        for i, line in enumerate(log):
            # fix windows and unix filepaths
            line[0] = os.path.join(img_folder, line[0].replace('\\','/').split('/')[-1])
            line[1] = os.path.join(img_folder, line[1].replace('\\','/').split('/')[-1])
            line[2] = os.path.join(img_folder, line[2].replace('\\','/').split('/')[-1])
            
            # take a random but contiguous chunk out of each dataset for train/validation split
            #   i.e. -> time -> |---------train---------|---valid---|----train----|
            if i >= valid_start_idx and i < valid_end_idx:
                valid_samples.append(line)
            else:
                train_samples.append(line)
    
    # Hyperparameters
    # for every sample, we have 3 images to train on (left, right, and center)
    train_samples_per_epoch = 3 * len(train_samples)
    valid_samples_per_epoch = 3 * len(valid_samples)
    batch_size = 32
    epochs = 2
    
    # initialize batch generators
    train_batch_gen = batch_gen(train_samples, batch_size=batch_size, shuffle=True, horizontal_flip=True)
    valid_batch_gen = batch_gen(valid_samples, batch_size=batch_size, shuffle=False, horizontal_flip=False)

    # compile, train, and save model
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_batch_gen, 
                        samples_per_epoch=train_samples_per_epoch, 
                        validation_data=valid_batch_gen,
                        nb_val_samples=valid_samples_per_epoch, 
                        nb_epoch=epochs)
    save_h5_model(model, args.model)
    return

def parse_args():
    parser = argparse.ArgumentParser(description='Remote Driving Model')
    parser.add_argument(
        'model',
        type=str,
        #required=True,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'log_folder',
        type=str,
        #required=True,
        help='Path to folder containing the folders for each log + associated batch of training images.'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)