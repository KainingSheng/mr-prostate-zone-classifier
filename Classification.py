"""
Created on Mon Nov  4 17:51:14 2019

@author: KainingSheng

"""

import os 
import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from dicomLoadUtils import csvLoader, dicomLoader
from preprocessingUtils import getLesionCoordinates, cropLesion, normalizer, resize2D
from postTrainingAnalysis import plotROC, getCIAUC

## Data Ingestion and Preprocessing
#Define path for data and csv-file for loading data. The default path links to the data-folder in this repository.
path = r'Data'

#Load in lists with patient IDs, T2-weighted sequence IDs, lesion coordinates (ijK), and the categorial zone labels for every patient from CSV file located in data folder
patient_id, sequence_id, ijk_t2w, zone = csvLoader(path,csv_file='loadValDataList.csv')

#Load in MR T2-weighted DICOM files and convert to arrays with 3-dimensions
t2w = dicomLoader(patient_id, sequence_id, path)

#Load in coordinates (xyz, provided in the csv file) for cropping out a 2D T2-weighted slice from the loaded 3D T2-volumes
x_t2w, y_t2w, z_t2w = getLesionCoordinates(ijk_t2w)

#Crop lesion Region of interest (ROI) from T2-weighted volumes based on provided coordinates in the csv. The ROI will have dimensions of 96x96x1
t2w_cropped = cropLesion(t2w,x_t2w,y_t2w,z_t2w,100,100)

#Resize to match 96x96, which is input format for the model. 
t2w_resized = resize2D(t2w_cropped, 96, 96)

#Normalize cropped ROI patches for training
t2w_norm = normalizer(t2w_resized)

#Defining testing data (X_test) and truth labels for comparison (y_test)
#Testing data must have 3 dimension in order to be compatible with the mode
X_test = np.expand_dims(t2w_norm, axis=-1)

#Testing labels must be encoded in order to accomodate the softmax output
encoder = LabelEncoder()
encoder.fit(zone)
y_test = list(encoder.transform(zone))

s = pd.Series(list(y_test))
y_test = pd.get_dummies(s)

##Loading test model
pre_trained_model = tf.keras.models.load_model(
    'Weights\\pcd_2D_zone_classifier4.h5',
    custom_objects=None,
    compile = True
    )

#Reporting accuracy and predictions of model
accuracy = pre_trained_model.evaluate(X_test, y_test, batch_size=4)
print('test loss, test acc:', accuracy)

predictions = pre_trained_model.predict(X_test, batch_size = 4)

#Preparing data for comparison the predicted values and the truth label in a ROC-curve
y_pred = [];
for i in range(len(predictions)):
    y_pred.append(predictions[i][1])
    y_truth = list(y_test[1])

#Reporting ROC AUC of model along with 95% Confidential Interval calculated based on bootstrapping 2000 samples
plotROC(y_truth,y_pred)
getCIAUC(y_truth,y_pred)

