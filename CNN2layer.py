# -*- coding: utf-8 -*-
"""
Created on Oct 24 15:36:30 2017

@author: Priyanka Mocherla
@version: 2.7

This code is for a two layer CNN to train and classify data of a continuous output. Reduced squared mean is 
used to calculate accuracy and loss. Separate validation data can be loaded and evaluated at the end of the training.
Activation values are saved and can be plotted with 'Hiddenlayer tsne.py'.

Functions in this module:
    - CNN
    
Please initialise these directories: 
    - trainingsaves/TSNE
    - model
"""

import numpy as np
from timeit import default_timer as timer
import tensorflow as tf

#--------------------- Change according to data format --------------------------

#Input/Output (Fixed to the dimensions of the waveform data)
NIN_width = 256
NIN_height = 4
OUT = 2
validate = False #Set to true if you have a validation dataset and scroll down to load the appropriate data.

#---------------------------- Load the data --------------------------------
load_start = timer()

waveforms = np.load("2018_01_25_waveforms_centre_ninepos.npy").reshape([-1,NIN_width*NIN_height])
labels = np.load("2018_01_25_labels_centre_ninepos.npy")

#-------------------------Setting Parameter Values------------------------------
#TUNABLE
batchsize = 32
steps = 1
filter_patch = [2,8]
FILTER1 = 8
FILTER2 = 16
DENSECON = 128
learning_rate = 0.001
save_step = 100
#strides = [1,2,8,1]

#------------------------ Shuffle and split the data ----------------------
randomise = np.arange(len(waveforms))
np.random.shuffle(randomise)
labels = labels[randomise]
waveforms = waveforms[randomise]

perc = 0.8 # Set the percentage of data that you want to split into test and train
train_set = waveforms[:int(len(waveforms)*perc)]
train_labels = labels[:int(len(waveforms)*perc)]
test_set = waveforms[int(len(waveforms)*perc):]
test_labels = labels[int(len(waveforms)*perc):]

#----------------------------- Set up validation data ------------------------------
if validate == True:
    validation_set = np.load('2018_01_15_waveforms_validations.npy').reshape([-1,NIN_width*NIN_height])
    validation_labels = np.load('2018_01_15_labels_validations.npy')

    randomise = np.arange(len(validation_set))
    np.random.shuffle(randomise)
    validation_labels = validation_labels[randomise]
    validation_set = validation_set[randomise]


load_end = timer()

print("%g seconds taken to prepare data" % (load_end - load_start))

#-----------------------Setting up the CNN---------------------------------
def weight_variable(shape, name):
    """ Initialises a weight tensor based on its shape and name
    
    Arguments:
    - shape: array, shape of the tensor to initialise
    - name: str, name of the initialised weight
    """
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape,name):
    """ Initialises a bias tensor based on its shape and name
    
    Arguments:
    - shape: array, shape of the tensor to initialise
    - name: str, name of the initialised bias
    """
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name =name)
    
def conv2d(x, W):
    """ Creates a 2D convolutional layer
    
    Arguments:
    - x: tensor, input tensor
    - W: tensor, weight tensor
    """
    return tf.nn.conv2d(x, W, strides = [1,2,8,1], padding = 'SAME')
    
def max_pool_2x2(x):
    """ Creates a 2x2 maxpooling layer
    
    Arguments:
    - x: tensor, input tensor
    """
    return tf.nn.max_pool(x, ksize=[1,1,1,1], strides = [1,1,1,1], padding = 'SAME')
    
def getActivations(layer,stimuli):
    """Calculates the activation values for a layer depending on a specified input waveform
    
    Arguments:
    - layer: tensor, the layer that should be activated (e.g. W_conv1)
    - stimuli: array, the input waveform used to activate the layer
    """
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,NIN_width*NIN_height],order='F'),keep_prob:1.0})
    return units

def layersTSNE(layer, no_events):
    """Produces a set of activations with corresponding inputs and labels for further t-SNE analysis.
    
    Arguments:
    - layer: tensor, the layer that should be activated
    - no_events: int, the number of sample points to use for t-SNE analysis
    """
    stimuli = waveforms[0:no_events]
    stimuli_labels = labels[0:no_events]
    test = getActivations(layer, stimuli[0])
    shape = test.shape
    
    if len(shape) == 4:
        unit_layers = np.empty([no_events, shape[1], shape[2], shape[3]])
        for i in range(no_events):
            unit_layers[i] = getActivations(layer, stimuli[i])
            
    if len(shape) == 2:
        unit_layers = np.empty([no_events, shape[1]])
        for i in range(no_events):
            unit_layers[i] = getActivations(layer, stimuli[i])

        
    return stimuli, stimuli_labels, unit_layers


#-------------------------Initialising variables --------------------------
x = tf.placeholder(tf.float32, shape=[None,NIN_height*NIN_width], name = 'x')
y_ = tf.placeholder(tf.float32, shape = [None, OUT], name = 'y_')
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(learning_rate)

#-------------------------- The CNN Model --------------------------------
#First Layer
W_conv1 = weight_variable([filter_patch[0],filter_patch[1],1,FILTER1], 'W_conv1')
b_conv1 = bias_variable([FILTER1], 'b_conv1')

input_layer = tf.reshape(x, [-1,NIN_width,NIN_height,1]) 

h_conv1 = tf.nn.relu(conv2d(input_layer, W_conv1)+b_conv1, name = 'h_conv1')
h_pool1 = max_pool_2x2(h_conv1)

#Second Layer
W_conv2 = weight_variable([filter_patch[0],filter_patch[1],FILTER1, FILTER2], 'W_conv2')
b_conv2 = bias_variable([FILTER2], 'b_conv2')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name = 'h_conv2')
h_pool2= max_pool_2x2(h_conv2)

#Densely Connected Layer
W_fc1 = weight_variable([(FILTER2*NIN_width*NIN_height)/16, DENSECON], 'W_fc1')
b_fc1 = bias_variable([DENSECON], 'b_fc1') 

h_pool2_flat = tf.reshape(h_pool2, [-1,(FILTER2*NIN_width*NIN_height)/16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1, name = 'h_fc1')

#Dropout
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#Output layer
W_fc2 = weight_variable([DENSECON,OUT],'Wfc2')
b_fc2 = bias_variable([OUT], 'b_fc2')
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Evaluation variables
cost = tf.reduce_mean(tf.square(y_conv-y_)) 
optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#--------------------- Initialise saved variable lists -------------------
testaccuracylist = []
trainaccuracylist = []
losslist = []

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep = 4)

#--------------------------- Run session --------------------------------
with tf.Session() as sess:
    run_start = timer()
    sess.run(tf.global_variables_initializer())
    #Restore the model
    #saver = tf.train.import_meta_graph('./model_1/run_2/run_1-9999.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('./model_1/run_2/'))
    
    for i in range(steps):
        randomise = np.arange(len(train_labels))
        np.random.shuffle(randomise)
        labelsbatch = train_labels[randomise][0:batchsize]
        waveformsbatch = train_set[randomise][0:batchsize]
        
        #Evaluate the model every 50 steps
        if (i+1) % save_step == 0: 
            test_accuracy = accuracy.eval( feed_dict={x: test_set, y_: test_labels, keep_prob: 1.0})
            train_accuracy = accuracy.eval(feed_dict={x: waveformsbatch, y_: labelsbatch, keep_prob: 1.0})
            _, c = sess.run([train_step,cost], feed_dict={x:waveformsbatch,y_:labelsbatch, keep_prob:0.5})
            print "Step %d: training accuracy %g, test accuracy %g, loss %g" % (i+1, train_accuracy, test_accuracy, c)
            losslist.append(c)
            testaccuracylist.append(test_accuracy)
            trainaccuracylist.append(train_accuracy)
            
            # Save some of the parameters every 10 steps
            y_conv_eval = y_conv.eval({x: test_set, y_: test_labels, keep_prob: 1.0})
            np.save('trainingsaves/step',i+1)
            np.save('trainingsaves/trainaccuracylist',trainaccuracylist)
            np.save('trainingsaves/testaccuracylist',testaccuracylist)
            np.save('trainingsaves/losslist',losslist)
            np.save('trainingsaves/finalevallabels',  y_conv_eval)
            np.save('trainingsaves/finaltestlabels',  test_labels)
            saver.save(sess, 'model/run_1', global_step=i)
            
        train_step.run(feed_dict = {x: waveformsbatch, y_: labelsbatch, keep_prob: 0.5})
        
    run_end = timer()
    print("Test accuracy %g" % accuracy.eval(feed_dict={x: test_set, y_: test_labels, keep_prob : 1.0}))
    
    if validate == True:
        y_val_eval = y_conv.eval({x: validation_set, y_: validation_labels, keep_prob : 1.0})
        np.save('trainingsaves/finalevalvalidationlabels',  y_val_eval)
        np.save('trainingsaves/finaltestvalidationlabels',  validation_labels)
        print("Validation accuracy %g" % accuracy.eval(feed_dict={x: validation_set, y_: validation_labels, keep_prob : 1.0}))
    
    print("%g minutes taken to complete run" % ((run_end - run_start)/60.))
    
    #------------------------ t-SNE data saving -----------------------------#
    n = 5000 #Number of training samples to use for t-SNE analysis
    np.save("trainingsaves/TSNE/wavTSNE.npy", layersTSNE(h_conv1, n)[0])
    np.save("trainingsaves/TSNE/labTSNE.npy", layersTSNE(h_conv1, n)[1])
    np.save("trainingsaves/TSNE/h_conv1TSNE.npy", layersTSNE(h_conv1, n)[2])
    np.save("trainingsaves/TSNE/h_conv2TSNE.npy", layersTSNE(h_conv2, n)[2])
    np.save("trainingsaves/TSNE/h_fc1TSNE.npy", layersTSNE(h_fc1, n)[2])
    
    #-------------------------  Layer/Weight saving -------------------------#
    imageToUse = waveforms[0]
    labelToUse = labels[0]
    
    #Loading activations of weights and hidden layers 
    unitsW1 = getActivations(W_conv1,imageToUse)
    unitsW2 = getActivations(W_conv2,imageToUse)
    unitsh1 = getActivations(h_conv1,imageToUse)
    unitsh2 = getActivations(h_conv2,imageToUse)
    unitsWfc2 = getActivations(W_fc2, imageToUse)
    evaluated = getActivations(y_conv, imageToUse)
    
    #Saving the activations
    np.save("trainingsaves/unitsW1.npy", unitsW1)
    np.save("trainingsaves/unitsW2.npy", unitsW2)
    np.save("trainingsaves/unitsh1.npy", unitsh1)
    np.save("trainingsaves/unitsh2.npy", unitsh2)
    np.save("trainingsaves/unitsWfc2.npy", unitsWfc2)
    np.save('trainingsaves/testimage.npy', imageToUse)
    np.save('trainingsaves/testlabel.npy', labelToUse)
    np.save('trainingsaves/predictedlabel.npy', evaluated)
    

