"""
Created on Oct 30 15:36:30 2017

@author: Priyanka Mocherla
@version: 2.7

This code is written to convert the data recieved from the lab computer to make sure it is in a suitable format for further analysis.
Run code through this to create the correctly shaped and sorted data which can be saved and directly input into the CNN. Further analysis code requires the data to be sorted.

Functions in this module:
    - add_padding
    - sorted_array
    - indexes
    - recentre
    - addZeros
"""

#---------------------------------- Functions ------------------------------------#
import numpy as np

def add_padding(waveforms, sample_size):
    """
        Returns the set of waveforms in a [Number of events, Number of channels, Number of samples] format. 
        Number of samples is set to 256 and number of channels is set to 4 by default.
    
        Args:
        waveforms - array, input waveforms 
        sample_size - int, the length desired for each event (DEFAULT : 512 for Am, 256 for Sr)           
    """
    #Initialise a waveform array in the desired format
    waveforms_new = np.empty([len(waveforms),4,sample_size])
    
    for i in range(len(waveforms)):
        for j in range(4):
            #Make sure that the [i][j] member of the events has exactly 256 entries
            if len(waveforms[i][j]) < sample_size:
                pad = 256 - len(waveforms[i][j])
                waveforms_new[i][j] = np.concatenate([waveforms[i][j], np.zeros(pad)]) #padding added 
            elif len(waveforms[i][j]) > sample_size:
                waveforms_new[i][j] = np.array(waveforms[i][j][0:sample_size]) #array sliced 
            else:
                waveforms_new[i][j] = np.array(waveforms[i][j])
    
    return waveforms_new
    
def sorted_array(waveforms, labels):
    """
        Returns the a set of waveforms sorted in label groups. 
        NOTE: the labels must be in np.arange([0.5, 5.5, 0.5])
    
        Args:
        waveforms - array, unsorted waveforms (padded)
        labels - array, unsorted labels           
    """
    #Initialising an empty array for the sorted labels and waveforms
    waveforms_new = np.empty([len(labels),4,waveforms.shape[-1]])
    labels_new = np.empty([len(labels),2])
    
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
        for j in range(4):
            waveforms_new[i][j] = waveforms_list[i][j]  
     
    
    return waveforms_new, labels_new


def indexes(sorted_labels):
    """
        Returns a set of indices for the end of each label group in the sorted data. 
    
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
    
def recentre(waveforms):
    """
        Returns the input waveforms recentred at sample 50
    
        Args:
        waveforms - array, padded waveforms
    """
    #Initialising the new recentred waveforms
    recentred_waveforms = np.empty([len(waveforms), 4, waveforms.shape[-1]])
    recentre_index = np.empty(len(waveforms))
    
    #Finding the channel containing the max amplitude and the index of where that max occurs.
    for i in range(len(waveforms)):
        maxes = np.zeros(4)
        max_index = np.zeros(4)
        for j in range(4):
            maxes[j] = waveforms[i][j].max()
            max_index[j] = np.where(waveforms[i][j] == maxes[j])[0][0]
  
        recentre_index[i] = int(max_index[maxes.argmax()])
    
    #Cut the waveform so that it is centred at 50 and pad accordingly.  
    for i in range(len(waveforms)):
        for j in range(4):
            padded = np.lib.pad(waveforms[i][j], (50,waveforms.shape[-1]), 'constant')
            recentred_waveforms[i][j] = padded[int(recentre_index[i]):int(recentre_index[i])+256]
          
    return recentred_waveforms
    
def addZeros(waveforms_array, back, forward):
    """
        Returns the input waveforms with specified areas before and after the peak set to zero
    
        Args:
        waveforms_array - array, waveforms
        back - int, the number of points before the peak you want to set to zero
        forward - int, the number of points after the peak you want to set to zero
    """
    #Finding the value and position of the largest peak in each event.
    for i in range(len(waveforms_array)):
        max_value, max_index = max((x, (i, j))
                              for i, row in enumerate(waveforms_array[i])
                              for j, x in enumerate(row))
        
        #Set values around the peak to zero
        for j in range(4):
            waveforms_array[i][j][0:max_index[1]-back] = 0 
            waveforms_array[i][j][max_index[1] + forward::] = 0
    
    return waveforms_array



# -------------------------- Example code -----------------------------#
#Load unsorted waveforms and labels from the server
waveforms = np.load("2018_02_14_waveforms.npy")
labels = np.load("2018_02_14_labels.npy")

padded_waveforms = add_padding(waveforms, 512) # MAKE SURE THIS IS UPDATED FOR THE SAMPLE LENGTH OF THE DATA
#recentred_waveforms = recentre(padded_waveforms)
#zero_waveforms = addZeros(recentred_waveforms, 5, 20)
sorted_data = sorted_array(padded_waveforms, labels)

sorted_waveforms = sorted_data[0]
sorted_labels = sorted_data[1]

#np.save("2018_02_14_waveforms_sorted.npy", sorted_waveforms)
#np.save("2018_02_14_labels_sorted.npy", sorted_labels)

