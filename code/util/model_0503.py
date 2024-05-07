import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from tensorflow.keras import models
from tensorflow.keras.applications.resnet50 import ResNet50

import tensorflow as tf
import os
from IPython.display import Image, display

import warnings
warnings.filterwarnings('ignore')

from skimage import io

import numpy as np
from PIL import Image
import random

from util.prev import dataset as da

# ---------------------------------------------------------------------------------

def model():

    initial_model = tf.keras.applications.VGG16(weights = 'imagenet',include_top = False)
    initial_model.trainable = False

    #adding all the layers from the architecture into a list
    layers = [l for l in initial_model.layers]

    #building a new model with the existing layers with their 'imagenet' weights
    model = keras.Sequential()
    inputs = keras.Input(shape = (688,96,3))
    model.add(inputs)

    #removing the last block from my VGG16 architecture and

    for layer in layers:
        if 'block5' not in layer.name:
            layer._inbound_nodes = []
            layer._outbound_nodes = []
            model.add(layer)


    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(1, activation="linear"))
    
    return model


# ---------------------------------------------------------------------------------

def half_dataset(data_num):

    X = np.load('save/data/X_688_96.npy',)
    y = np.load('save/data/y_decrease.npy')


    # Original Dataset

    X1 = X[3:56]
    y1 = y[3:56]

    X2 = X[83:160]
    y2 = y[83:160]

    X3 = X[162:231]
    y3 = y[162:231]

    X4 = X[235:312]
    y4 = y[235:312]


    for i in range(len(X3)):
        X3[i][:6,:,:] = 1
        
# -----------------------
    if data_num == 0:
        X_train = np.concatenate( [ X1, X2, X3] )
        X_test = np.concatenate(  [ X4]  )

        y_train = np.concatenate( [ y1,y2,y3] )
        y_test = np.concatenate( [ y4 ] )


    elif data_num == 1:
        X_train = np.concatenate( [X1, X2, X4 ] )
        X_test = np.concatenate(  [ X3 ]  )

        y_train = np.concatenate( [ y1, y2, y4 ] )
        y_test = np.concatenate( [ y3 ] )
        
    elif data_num == 2:
        X_train = np.concatenate( [X1, X3, X4 ] )
        X_test = np.concatenate(  [ X2 ]  )

        y_train = np.concatenate( [ y1, y3, y4 ] )
        y_test = np.concatenate( [ y2 ] )
        
    elif data_num == 3:
        X_train = np.concatenate( [X2, X3, X4 ] )
        X_test = np.concatenate(  [ X1 ]  )

        y_train = np.concatenate( [ y2, y3, y4 ] )
        y_test = np.concatenate( [ y1 ] )
        
    return str(data_num), (X_train, y_train), (X_test, y_test)

# ---------------------------------------------------------------------------------        

def train(model_, weight, epoch, num, X_train, y_train):
    
    train_mse_metric_p = keras.metrics.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    previous_list = [100]
    l1_penalty = 0
     
    violation_track = np.array([0,0,0,0,0,0,0])
    
    #for i in range(5):  
    for i in range(len(y_train)):   
        with  tf.GradientTape() as tape:

            image = np.expand_dims(X_train[i], axis=0)
            y_hat = model_(image, training=True)

            previous_value = previous_list[-1]
            violation_term = tf.constant(max( (y_hat - previous_value) , 0), dtype=tf.float32)

            if i ==0: violation_term = 0
            if i !=0 and  float(y_train[i]) > float(y_train[i-1]):
                violation_term = 0

            y_train_c = tf.constant(y_train[i])

            mse = tf.keras.losses.MeanSquaredError()(y_train_c, y_hat)
            penalty = violation_term  * weight

            # calculate loss
            loss_value =  mse + penalty
            previous_list.append(y_hat)

        grads = tape.gradient(loss_value, model_.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_.trainable_weights))
            
        # Update training metric.
        train_mse_metric_p.update_state(y_train[i], y_hat)
        
        l1_penalty = (l1_penalty + penalty) 
        
        #if epoch % 5 == 0 and violation_term != 0: 
        if violation_term !=0:
            
            violation_track = np.vstack( (violation_track , [epoch, y_train[i], tf.get_static_value(y_hat)[0][0], mse, violation_term[0][0], penalty[0][0], num] ) )
                
    # Display metrics at the end of each epoch.
    train_mse = train_mse_metric_p.result().numpy()
    
    l1_penalty = l1_penalty / len(y_train)
    
    violation_track = np.delete(violation_track, (0), axis=0)
    

    return model_, train_mse, l1_penalty, violation_track

# ---------------------------------------------------------------------------------

def test(model_, X_test, y_test):
    
    # before new training
    test_mse_metric_p = keras.metrics.MeanSquaredError()
  
    
    #for i in range(5):   
    for i in range(len(y_test)):
        image = np.expand_dims(X_test[i], axis=0)
        y_hat = model_(image, training=False)

        y_test_c = tf.constant(y_test[i])
        mse = tf.keras.losses.MeanSquaredError()(y_test_c, y_hat)

        # Update training metric.
        test_mse_metric_p.update_state(y_test[i], y_hat)

    # Display metrics at the end of each epoch.
    test_mse = test_mse_metric_p.result().numpy()
    
    return test_mse

# ---------------------------------------------------------------------------------

def full_half_training(data_num,model, epochs, weight):
    
    array = np.array([0,0,0,0,0,0,0])
    
    list_train_mse = []
    list_penalty_mse = []
    list_test_mse = []
    list_num = []
    
    for epoch in range(epochs+1):
        
        
        num, (X_train, y_train), (X_test, y_test) = half_dataset(data_num) 

        model, train_mse, penalty, violation_track = train(model, weight,epoch, int(num), X_train, y_train)
        test_mse = test(model, X_test, y_test)
        
        array = np.vstack( (array , violation_track ) ) # need to be changed number according to the stack condition in train function.
        
        list_train_mse.append(train_mse)
        list_penalty_mse.append(penalty)
        list_test_mse.append(test_mse)
        
        
    array = np.delete(array, (0), axis=0)
    
    df = pd.DataFrame(array, columns = ['epochs', 'y true', 'y_hat', 'mse', 'violation_term','penalty(v*w)', 'test dataset'])
    
    
    # list_train_mse and list_train_mse1 / list_test_mse and list_test_mse1 should be same
    loss = np.array([list_train_mse, list_penalty_mse, list_test_mse])
    
    epochs = epochs + 100
    
    weight_s = str(weight)
    epochs_s = str(epochs)

    df.to_csv('save/excel/w'+weight_s + '_e' + epochs_s  +'_n'+ num+'_half' + '.csv')
    model.save_weights('save/model/w'+ weight_s + '_e' + epochs_s + '_n'+ num+'_half' +'.h5')
    np.save('save/w'+ weight_s + '_e' + epochs_s  + '_n'+ num +'_half'+'.npy', loss)