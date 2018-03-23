"""
@author: Priyanka Mocherla
@version: 2.7

This code is for a two layer CNN to plot the activations values obtained from an input waveform
for specified layers.

Functions in this module:
    - plotweights
    - plothidden
"""

import numpy as np
import math as math
import matplotlib.pyplot as plt

# -------------------------- Plotting functions ----------------------------#
def plotweights(units, title):
    """ Plots the weights of each filter for a specific layers activation values
    
    Arguments:
    - units: array, the values of the activation for the layer
    - title: str, the title of the plot
    """
    #Plots the weights of the convolutional layers
    if len(units.shape) == 4:
        filters, inputs = units.shape[3], units.shape[2]
        n_columns = 4
        n_rows = math.ceil(filters / n_columns) + 1
        for ninputs in range(inputs):
            plt.figure()
            plt.suptitle(str(title) + " - Input Channel " +str(ninputs+1))
            for i in range(filters):
                plt.subplot(n_rows, n_columns, i+1)
                plt.title('Filter ' + str(i+1))
                weights = []
                for height in range(units.shape[0]):
                    for width in range(units.shape[1]):
                        weights.append(units[height][width][ninputs][i])
                
                plt.imshow(np.array(weights).reshape(units.shape[0],units.shape[1]), interpolation="nearest", cmap="hot")
    
    #Plots the weights of the fully connected layers i.e. the output activations      
    else:
        #Labels for each activation layer
        lab = ['x', 'y', 'z']
        
        #Plots the activation values of each node for each output
        for i in range(units.shape[1]):
            plt.figure()
            toPlot = []
            for j in range(units.shape[0]):
                toPlot.append(units[j][i])
            
            plt.plot(toPlot, 'o-')
            plt.xlabel("Node")
            plt.ylabel("Activation value")
            plt.title(str(title) + " - " + str(lab[i]))
        
    plt.show()
    
def plothidden(units, title):
    """ Plots the weights of each filter for a specific channel
    
    Arguments:
    - units: array, the values of the activation for a layer
    - title: str, the title of the plot
    """    
    filters, inputs = units.shape[3], units.shape[2]
    n_columns = 4
    n_rows = math.ceil(filters / n_columns) + 1
    
    #Plots the hidden layers for a given set of inputs
    for ninputs in range(inputs):
        plt.figure()
        plt.suptitle(str(title))
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i+1)
            plt.title('Filter ' + str(i+1))
            weights = []
            for height in range(units.shape[0]):
                for width in range(units.shape[1]):
                    weights.append(units[height][width][ninputs][i])
            plt.imshow(np.array(weights).reshape([4,units.shape[1]/4]), interpolation="nearest", cmap="hot")
    plt.show()
    
# -------------------------- Loading the data ------------------------------#
unitsW1 = np.load("trainingsaves/unitsW1.npy")
unitsW2 = np.load("trainingsaves/unitsW2.npy")
unitsh1 = np.load("trainingsaves/unitsh1.npy")
unitsh2 = np.load("trainingsaves/unitsh2.npy")
unitsWfc2 = np.load("trainingsaves/unitsWfc2.npy")
imageToUse = np.load('trainingsaves/testimage.npy')
labelToUse = np.load('trainingsaves/testlabel.npy')
evaluated = np.load('trainingsaves/predictedlabel.npy')
    
# -------------------------- Plotting the data ------------------------------#
    
#Plotting the filter weights and hidden layer activations
plotweights(unitsW1, "Layer 1 filters")
plotweights(unitsW2, "Layer 2 filters")
plothidden(unitsh1, "Hidden Layer 1")
plothidden(unitsh2, "Hidden Layer 2")
plotweights(unitsWfc2, "Output Activations")

#Plotting the waveform that is used for the following visualisations.
plt.figure()
plt.title('Input test image')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plot = imageToUse.reshape([4,-1])
for i in range(4):
    plt.plot(plot[i], label = str(i+5))
plt.legend()
plt.show()

print('Test output: ' + str(evaluated) + ', True output: ' + str(labelToUse))