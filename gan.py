# -*- coding: utf-8 -*-
"""
Generative Adversarial Network

This is an example of GAN, applied in the field of fundamental data and time
series, even if not based on Recurrent cells. The objective is to train a network
that can generate a realistic set of fundamental data, to train another algorithm.

/ Comments
* The structure of the NN is pre determined, so users need to modify the code to 
adapt to their problems, so it can be easily called by the command prompt.
The fundamental data shape is (None,10)
* During training, the algorithm will generate 3 models, which are going to be 
  saved in the folder:
    scaler
    generator
    discriminator
* The algorithm keeps the best pair models, testing the accuracy in the training 
set and in the validation set, and gives the same weight to both.

-- Generator
The structure is the following:
Input(3) - Dense(10) - Dense(20) - Dense(10)

-- Discriminator
The structure is the following:
Input(10) - Dense(20) - Dense(10) - Dense(1)

The file has also been designed to be called with the command prompt
> cd 'directory'
> python gan.py --mode train --data 'name of input' -- epochs 'xxx'
> python gan.py --mode generate --output 'name of output'

Command inputs:
    --mode: train / generate
    --data: input name
    --epochs: number of epochs for the GAN to be ran through
    --batch_size: batch size of runs
    --output: output name
"""

# Import libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Activation
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score

import numpy as np

import os

import argparse

CHEMIN = 'C:\\Users\\pages\\Desktop\\Work\\'

os.chdir(CHEMIN)

def generator_model():
    """Generator model building. The structure is already pre-determined, and this
    can be modified manually by the user"""
    
    # Layer 1
    model = Sequential()
    model.add(Dense(10, input_dim = 3))
    model.add(Activation('tanh'))
    
    # Layer 2
    model.add(BatchNormalization())
    model.add(Dense(20))
    model.add(Activation('tanh'))
    
    # Layer 3
    model.add(Dense(10))
    model.add(Activation('tanh'))
    
    return model
    
def discriminator_model():
    """Discriminator model building. The structure is already pre-determined, 
    and this can be modified manually by the user"""
    
    # Layer 1
    model = Sequential()
    model.add(Dense(20, input_dim = 10))
    model.add(Activation('tanh'))
    
    # Layer 2
    model.add(Dense(10))
    model.add(Activation('tanh'))
    
    # Layer 3
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

def generator_containing_discriminator(g, d):
    """Combining both generator and discriminator models"""
    
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    
    return model

def random_split(X):
    """Randomly split the matrix X, keeping 80% of the data for training and
    20% for testing"""
    
    # Select 80% of the data
    lim = int(X.shape[0] * 0.8)
    
    # Random permutations
    X = X[np.random.permutation(X.shape[0]), :]
    
    return X[:lim,:], X[lim:,:]

def train(X, BATCH_SIZE, EPOCHS):
    """Train the GAN network"""
    
    # To keep accuracy metric
    err = []
    eps = np.inf
    
    # Extract the train / test information
    X_train, X_test = random_split(X)
    
    # Transform the raw input with a StandardScaler(), and save its parameters
    s = StandardScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)
    joblib.dump(s, 'scaler.sav')
    
    # Initialise the generator and the discriminator
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr = 0.0005, momentum = 0.9, nesterov = True)
    g_optim = SGD(lr = 0.0005, momentum = 0.9, nesterov = True)
    g.compile(loss = 'mean_squared_error', optimizer = 'SGD')
    d_on_g.compile(loss = 'binary_crossentropy', optimizer = g_optim)
    d.trainable = True
    d.compile(loss = 'binary_crossentropy', optimizer = d_optim)
    
    # Go through epochs
    for epoch in range(EPOCHS):
        
        # Every 10 epoch print the follow up
        if epoch % 100 == 0:
            print('Epoch {} / {}'.format(epoch, EPOCHS))
        
        # Through batches: numpy slicing will limit to shape of X_train
        for index in range(int(X_train.shape[0] / BATCH_SIZE)+1):
            data_batch = X_train[index * BATCH_SIZE : (index+1) * BATCH_SIZE]
            TAILLE = data_batch.shape[0]
            
            noise = np.random.uniform(-1, 1, size = (TAILLE,3))
            
            data_generated = g.predict(noise, verbose = 0)
            
            # Variables to be distinguished for the batch run
            xx = np.concatenate((data_batch, data_generated))
            y = [1] * TAILLE + [0] * TAILLE
            
            # Train the discriminator
            d.train_on_batch(xx, y)

            # Train the generator
            noise = np.random.uniform(-1, 1, size = (TAILLE,3))
            d.trainable = False
            d_on_g.train_on_batch(noise, [1] * TAILLE)
            d.trainable = True
            
        # Check the pertinence of the model after each epoch, testing on both
        # the training and test sets
        acc = model_accuracy(X_train, X_test, g, d)
        
        # Keep accuracy metric
        err.append(acc)
        
        # Keep best model
        if np.abs(0.5 - acc) < eps:
            print('Model saved at epoch {} with accuracy {}'.format(epoch, acc))

            eps = np.abs(0.5 - acc)

            # Save weights
            g.save_weights('generator', True)
            d.save_weights('discriminator', True)
        
    return err
        
def model_accuracy(X_train, X_test, g, d):
    """Function computing a single metrics of the GAN approach, to determine 
    if we save the model or not"""

    # Generate fake data
    noise_train = np.random.uniform(-1, 1, size = (X_train.shape[0], 3))
    noise_test = np.random.uniform(-1, 1, size = (X_test.shape[0], 3))    
    
    data_generated_train = g.predict(noise_train, verbose = 0)
    data_generated_test = g.predict(noise_test, verbose = 0)
        
    y_1a = d.predict_classes(data_generated_train)
    y_1b = d.predict_classes(X_train)
    
    y_2a = d.predict_classes(data_generated_test)
    y_2b = d.predict_classes(X_test)
    
    train_accuracy = (accuracy_score(np.zeros(X_train.shape[0]), y_1a) + 
                      accuracy_score(np.ones(X_train.shape[0]), y_1b)) / 2
    test_accuracy = (accuracy_score(np.zeros(X_test.shape[0]), y_2a) + 
                      accuracy_score(np.ones(X_test.shape[0]), y_2b)) / 2    
        
    return (train_accuracy + test_accuracy) / 2
                     
def generate(BATCH_SIZE):
    """Generate data"""
    
    # Load generator data, as saved
    g = generator_model()
    g.compile(loss = 'mean_squared_error', optimizer = 'SGD')
    g.load_weights('generator')
    
    noise = np.random.uniform(-1, 1, size = (BATCH_SIZE,3))
    data_generated = g.predict(noise, verbose = 0)
    
    return data_generated

def load_input(name):
    """Function loading the input variables from the folder"""
    
    return np.load(name)

def get_args():
    """Command prompt"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str)
    parser.add_argument('--data', type = str)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--output', type = str, default = 'data_new.npy')
    args = parser.parse_args()
    return args

# Main part, launched with the command prompt
if __name__ == '__main__':
    args = get_args()
    
    if args.mode == 'train':
        # If train, then load inputs and train the network
        X = load_input(args.data)
        _ = train(X, BATCH_SIZE = args.batch_size, EPOCHS = args.epochs)
        
    elif args.mode == 'generate':
        # If we want data generation
        X_new = generate(BATCH_SIZE = args.batch_size)
        s = joblib.load('scaler.sav')
        X_new = s.inverse_transform(X_new)
        np.save(args.output, np.array(X_new))
        
    
            
            
            
            
            
            
            
            
    
    
    