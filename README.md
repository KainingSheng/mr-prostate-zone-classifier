# MRI prostate exam zone classifier
This is a 2D Convolutional Neural Network for classifying the location of prostate gland lesions found in T2-weighted MRI images. The model takes in MRI dicom files (.dcm) along with coordinates (ijk) for the suspected lesion. It then classifies the lesions as either belong to the periferal zone (PZ), transitional zone (TZ) or anterior fibromuscular stroma (AS). 

Detailed instructions for usage: 

1) Download and extract the whole repository to your local machine

2) In the 'Data' folder, put in the T2-weighted MRI images you wish classified. As an example, a few MRI dataset from a publicly available dataset (ProstateX) is already included. Remove those before you put in your own data. The directories in your data should be organized in the following manner for each patient scan: patient_id -> sequence_id -> dicomfile.dcm

3) Data is loaded into the program through a csv-file, which is also loacted in the 'Data' folder. Put in your own csv-file with the following format: 
  row1: patient_id, 
  row2: sequence_id, 
  row3: lesion coordinates as image col, row and slice number. Col and row are what pixel values in the x and y direction the lesion is       located at, respectively, while slice number is the n'th slice with the most cranial slice being slice zero,  
  row4: zone label (either PZ, TZ or AS), 
See the existing csv-file (loadDataList.csv) for reference

4) From the Google Drive link provided in the 'Weight' folder, download the pre-trained model weights and put the .h5 file in the 'Weight' folder on your local machine. 

5) Run the 'Classification.py' script. It should report predictions for your dataset together with overall accuracy measures for the predictions. 
