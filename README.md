# MRI prostate exam zone classifier
This is a 2D Convolutional Neural Network for classifying the location of prostate gland lesions found in T2-weighted MRI images. It is a crucial first step in the development of fully autonomous prostate cancer significance classifiers. 

Detailed instructions for usage: 

1) Download the whole repository to your local machine

2) In the 'Data' folder, put in the T2-weighted MRI images you wish classified. As an example, a few MRI dataset from a publicly available dataset (ProstateX) is already included. Remove those before you put in your own data. The directories in your data should be organized in the following manner for each patient scan: patient_id -> sequence_id -> dicomfile.dcm

3) From the Google Drive link provided in the 'Weight' folder, download the pre-trained model weights and put the .h5 file in the 'Weight' folder on your local machine. 

4) Run the 'Classification.py' script. It should report predictions for your dataset together with overall accuracy measures for the predictions. 
