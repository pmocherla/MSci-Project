"""
@author: Priyanka Mocherla
@version: 2.7

This code is used to create features for each event using the input waveforms

Functions in this module:
    - integral
    - amplitude
    - peakCounter
    - xlabel
    - ylabel
    - featureSpace
    
Notes:
    The waveforms should be in a [No. events, 4, No. samples] format and the labels should be [No. events, 2] format. (Only supports 2D positions)
"""

#-------------------------------Functions------------------------------#
import numpy as np

def integral(waveforms, crop = False):
    """
        Returns the integrals of each channel from each event in a [No. events, 4] shape array.
    
        Args:
        waveforms - array, input waveforms 
        crop - bool, if True will crop the data within specified limits, only useful for single peak data         
    """
    integrals = np.empty([len(waveforms),4])
    
    for i in range(len(waveforms)):
        if crop == False:
            for j in range(4):
                integrals[i][j] = np.sum(waveforms[i][j])
        
        else:
            ind =  np.where(waveforms[i][3] == max(waveforms[i][3]))[0][0]
            for j in range(4):
                integrals[i][j] = np.sum(waveforms[i][j][ind-5:ind + 50])
                
    return integrals
    
def amplitude(waveforms):
    """
        Returns the amplitudes of each channel from each event in a [No. events, 4] shape array.
    
        Args:
        waveforms - array, input waveforms           
    """
    amplitudes = np.empty([len(waveforms),4])
    
    for i in range(len(waveforms)):
        for j in range(4):
            amplitudes[i][j] = np.max(waveforms[i][j])
            
    return amplitudes

def peakCounter(waveforms, thresh = 40):
    """
        Returns number of peaks in waveform.
    
        Args:
        waveforms - array, input waveforms 
        thresh - int, a spike threshold related to the gain of the system, set to 40 for these datasets.          
    """
    
    output = np.zeros([len(waveforms),4])
    
    for i in range(len(waveforms)):
        for j in range(4):
            counter = 0
            samples = waveforms[i][j]
            for k in range(len(samples)-2):
                    if samples[k+1] >= thresh and samples[k+1] > samples[k] and samples[k+1] > samples[k+2]:
                            counter += 1
            output[i][j] = counter
    
    return output
    
def correlation(waveforms, correlated_by = 'amp'):
    """
        Returns correlations between channels in the waveform
    
        Args:
        waveforms - array, input waveforms 
        correlated_by - 'int' or 'amp', will find correlations in the data based on the integer or amplitude value          
    """
    correlation = np.empty([len(waveforms),6])
    n = 1.
    
    if correlated_by == 'amp':
        amp = amplitude(waveforms)
        correlation[:,0] = amp[:,0]*amp[:,1]/n
        correlation[:,1] = amp[:,0]*amp[:,2]/n
        correlation[:,2] = amp[:,0]*amp[:,3]/n
        correlation[:,3] = amp[:,1]*amp[:,2]/n
        correlation[:,4] = amp[:,1]*amp[:,3]/n
        correlation[:,5] = amp[:,2]*amp[:,3]/n
            
    if correlated_by == 'int':
        inte = integral(waveforms)
        correlation[:,0] = inte[:,0]*inte[:,1]/n
        correlation[:,1] = inte[:,0]*inte[:,2]/n
        correlation[:,2] = inte[:,0]*inte[:,3]/n
        correlation[:,3] = inte[:,1]*inte[:,2]/n
        correlation[:,4] = inte[:,1]*inte[:,3]/n
        correlation[:,5] = inte[:,2]*inte[:,3]/n
        
    return correlation
            
    
def xlabel(labels):
    """
        Returns the x coordinate label of the labels
    
        Args:
        labels - array, data labels         
    """
    x = np.array(labels[:,0])
    return x.reshape([len(labels),1])
    
def ylabel(labels):
    """
        Returns the y coordinate label of the labels
    
        Args:
        labels - array, data labels         
    """
    y = np.array(labels[:,1])
    
    return y.reshape([len(labels),1])

def featureSpace(waveforms, labels):
    """
        Returns an array of desired variables in feature space
    
        Args:
        waveforms - array, input waveforms
        labels - array, labels of the waveforms            
    """
    amps = amplitude(waveforms) #
    ints = integral(waveforms, True)
    #peaks = peak_counter(waveforms,  40)  #(to add this put *peaks[i]* in line 101)
    #correlations = correlation(waveforms)
    
    x = xlabel(labels)
    y = ylabel(labels)
    
    features = np.empty([len(waveforms), 10]) #update this
    
    for i in range(len(waveforms)):
        features[i] = np.concatenate([amps[i], ints[i], x[i], y[i]])
        
    return features
    
#---------------------- Loading and saving the data ------------------------------# 
waveforms = np.load("2018_01_25_waveforms_sorted_ninepos.npy")
labels = np.load("2018_01_25_labels_sorted_ninepos.npy")

a = featureSpace(waveforms, labels)
np.save("2018_01_25_features_ninepos.npy", a)