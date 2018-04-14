# MSci-Project
Code related to my Master's project to use deep learning to improve the resolution of a particle detector. Took waveform data produced from a particle detector and used machine learning algorithms to verify whether the CNN/DF could reconstruct the position of the particle within the detector.

ActivationsPlotter - Plots the activations of a hidden layer in the CNN.
CNN2layer - The CNN code used for the project. It runs the network and saves the data for further analysis.
DecisionForest - The DF code used for the project. Runs feature data through and saves data for further analysis.
DecisionTreeFeatureCompiler - Converts the raw data into its key features including: number of peaks, amplitude, integral
Hiddenlayer tsne - Performs a t-SNE analysis on data from the hidden layers of CNN. Also is able to return a subset of the waveforms for a specified region in the t-SNE plot.
PreprocessingData - Processes the raw data by padding, sorting and centering them, for further preliminary/machine learning analysis.
