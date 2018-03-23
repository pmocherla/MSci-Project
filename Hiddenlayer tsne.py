"""
@author: Priyanka Mocherla
@version: 2.7

This code is used to plot the cumulative max integral and amplitude for each channel for each position.

Functions in this module:
    - flatten
    - sorted_array
    - indexes
    - plotter
    - range_plotter
    
Notes:
    The waveforms should be in a [No. events, 4, No. samples] format and the labels should be [No. events, 2] format.
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math

#---------------------------- Load the data ---------------------------# 
h_conv1 = np.load('trainingsaves/TSNE/h_conv1TSNE.npy')
h_conv2 = np.load('trainingsaves/TSNE/h_conv2TSNE.npy')
h_fc1 = np.load('trainingsaves/TSNE/h_fc1TSNE.npy')
labTSNE = np.load('trainingsaves/TSNE/labTSNE.npy')
wavTSNE = np.load('trainingsaves/TSNE/wavTSNE.npy') 

#------------------------------ Functions -----------------------------#
def flatten(layer):
    """
        Returns a flattened layer activations.
        
        Args:
        layer - tensor, activation values of a single input layer
    """
    n_points = layer.shape[0]
    n_dims = np.prod(layer.shape[1::])
    
    flat = np.empty([n_points, n_dims])
    
    for i in range(layer.shape[0]):
        flat[i] = layer[i].reshape([1,n_dims])
        
    return flat
    
def sorted_array(waveforms, labels):
    """
        Returns the a set of waveforms sorted in label groups. 
        NOTE: the labels must be in np.arange([0.5, 5.5, 1.0])
    
        Args:
        waveforms - array, unsorted input waveforms 
        labels - array, unsorted waveforms labels          
    """
    #Initialising an empty array for the sorted labels and waveforms
    waveforms_new = np.empty(waveforms.shape)
    labels_new = np.empty(labels.shape)
    
    #Intermediate lists for sorting
    waveforms_list = []
    labels_list = []

    #Looping through possible coordinates to group labels
    for i in np.arange(0.5,5.5,0.5):
        for j in np.arange(0.5,5.5,0.5):
            for k in range(len(labels)):
                if labels[k][0] == i:
                    if labels[k][1] == j:
                        labels_list.append(labels[k])
                        waveforms_list.append(waveforms[k])

    #Filling items into arrays            
    for i in range(len(labels)):
        labels_new[i] = labels_list[i]
        waveforms_new[i] = waveforms_list[i]  
     
    return waveforms_new, labels_new
    
def indexes(sorted_labels):
    """
        Returns a set of indices for each label group in the sorted data. 
    
        Args:
        sorted_labels - array, sorted labels             
    """
    indexes = {}

    #Counting the number of labels in each group
    for i in range(len(sorted_labels)):
        if str(sorted_labels[i]) not in indexes.keys():
            indexes[str(sorted_labels[i])] = 1
        else:
            indexes[str(sorted_labels[i])] += 1
    
    #Counting the index of each group
    for i in range(len(indexes)-1):
        keys = sorted(indexes.keys())
        indexes[keys[i+1]] = indexes[keys[i]] + indexes[keys[i+1]]
            
    return indexes


def plotter(layer, labels, title):
    """
        Produces a t-SNE plot of the input activations of a single layer
        
        Args:
        layer - array, activations values of the layer for a set of events
        labels - array, labels of the layer activations
        title - str, title of the plot
    """
    layer = flatten(layer)
    sorted_layer, labels = sorted_array(layer, labels)
    indexes_dict = indexes(labels)
    keys, values = zip(*sorted(zip(list(indexes_dict.keys()), list(indexes_dict.values()))))
    values = [0] + list(values)
    
    #Run the t-SNE
    TSNElayer = TSNE(n_iter = 2000, n_components=2, verbose = 2, perplexity = 30, random_state = 0).fit_transform(sorted_layer)
    
    #Plot the data
    plt.figure()
    for i in range(len(keys)):
        plt.plot(TSNElayer[:,0][values[i]:values[i+1]], TSNElayer[:,1][values[i]:values[i+1]], '.', label = str(keys[i]))
    
    plt.xlabel("t-SNE [x]")
    plt.ylabel("t-SNE [y]")    
    plt.title(str(title))
    plt.legend()  
    plt.show()

    return TSNElayer
    
def range_plotter(TSNElayer, x, y, max_seen = 9, waveforms = wavTSNE, labels = labTSNE):
    """
        Returns the plots waveforms of points within a certain range after t-SNE is performed
        
        Args:
        TSNElayer - array, the activation values of the sample of events to perform t-SNE on
        x - list, start and end x values to scan t-SNE plot.
        y - list, start and end y values to scan t-SNE plot.
        max_seen - int, the number of points you want to return within the specified range (These will plotted so ideally keep this < 20)
        waveforms - array, corresponding input waveforms for each layer activation event
        labels - array, label of each waveform
    """
    #Shuffle the data 
    sorted_waveforms, sorted_labels = sorted_array(waveforms,labels)
    randomise = np.arange(len(sorted_waveforms))
    np.random.shuffle(randomise)
    sorted_labels = sorted_labels[randomise]
    sorted_waveforms = sorted_waveforms[randomise]
    TSNElayer = TSNElayer[randomise]
    
    #Initialise plotting variables
    counter = 0
    n_columns = 3
    n_rows = math.ceil(max_seen / n_columns) + 1
    
    #Plotting
    plt.figure()
    for i in range(len(TSNElayer)):
        if TSNElayer[i][0] >= x[0] and TSNElayer[i][0] <= x[1] and TSNElayer[i][1] >= y[0] and TSNElayer[i][1] <= y[1]:
            wav = sorted_waveforms[i].reshape([4,256])
            counter = counter + 1
            print TSNElayer[i]
            
            plt.subplot(n_rows, n_columns, counter)
            plt.title(str(sorted_labels[i]))
            for k in range(4):
                plt.plot(wav[k])
            
        if counter == max_seen:
            break
            
    plt.subplots_adjust(hspace=1)
    plt.show()
    
    return counter
             

#--------------------------- Example Code --------------------------#  
test = plotter(h_fc1, labTSNE, "FC1")

#To view some sample waveforms, run this function in the interactive kernel. Can adjust the range according to the points you want to see from the t-SNE plot.
"""
x = [-20,-15]
y = [10, 20]
range_plotter(test, x, y, max_seen = 9, waveforms = wavTSNE, labels = labTSNE)
"""
