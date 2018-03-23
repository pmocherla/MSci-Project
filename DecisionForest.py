"""
@author: Priyanka Mocherla
@version: 2.7

This code runs a decision forest on a dataset of features

Functions in this module:
    - Decision Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Set the coordinate you are analysing and the coordinate you are dropping here (For the side please note that the settings should correspond to y = 'x' and z = 'y')
coord_a = 'x'
coord_d = 'y'

#--------------------------- Setting up the model -----------------------------#
sns.set(style="white", font_scale=0.9)

#Load and create the dataset
dataset_channels = np.load("2018_01_25_features_ninepos.npy")
dataset = pd.DataFrame(dataset_channels, columns = ["Amp5", "Amp6", "Amp7", "Amp8", "Int5", "Int6", "Int7", "Int8","xlabel", "ylabel"]) #Add columns according to all the features included in the data

#Create y and X variables
y = dataset[str(coord_a)+"label"]
X = dataset.drop(str(coord_a)+"label", axis=1)

#Feature scaling (and dropping the ylabel because we dont want it as a feature at the moment)
sc = StandardScaler()

X_scaled = X.drop(str(coord_d)+"label", axis = 1)
X_scaled = pd.DataFrame(sc.fit_transform(X_scaled), columns = X_scaled.columns.values)

#Splitting the data into train and test set 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

#--------------------------------- Running the model ---------------------------------#
#Random Forest Regression
regressor = RandomForestRegressor(random_state = 0, n_estimators = 250, max_depth = 30, max_features = 5, min_samples_leaf = 25, min_samples_split = 10, bootstrap = True, verbose = 3)
regressor.fit(X_train, y_train)
print(regressor.score(X_test, y_test))


y_pred = regressor.predict(X_test)
print(mean_squared_error(y_test, y_pred))
plt.title("2D Prediction Distribution")
plt.hist2d(x = y_test, y = y_pred)
plt.colorbar()
plt.xlabel("True " + str(coord_a) + "label")
plt.ylabel("Predicted " + str(coord_a) + "label")
plt.show()

#I---------------------------- Interpreting the model ------------------------------#
plt.figure()
feature_import = pd.DataFrame(data = regressor.feature_importances_, index = X_scaled.columns.values, columns = ['values'])
feature_import.sort_values(['values'], ascending = False, inplace = True)
feature_import.transpose()
feature_import.reset_index(level=0, inplace=True)
plt.title("Feature Importance")
sns.barplot(x='index', y='values', data=feature_import)
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.show()

#-----------------------Saving test and prediction labels-------------------------#
np.save(str(coord_a) + "_pred2", y_pred)
np.save(str(coord_a) + "_test2", y_test)