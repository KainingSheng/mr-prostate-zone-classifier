# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:14:31 2019

@author: kaikai
"""

# import SimpleITK as sitk
import numpy as np
import pydicom
import os
import csv

def csvLoader(csv_path,csv_file):
    with open(csv_path + '\\' + csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_file, None)
    
        patient_id=[]
        sequence_t2w=[]
        ijk_t2w = []
        zone = []
        zone_label = []
        
        for row in csv_reader:
            patient_id.append(row[0])
            sequence_t2w.append(row[1])
            ijk_t2w.append(row[2])
            zone.append(row[3])
            zone_label.append(row[4])
    print('Patients included in list:', len(patient_id))
    print('T2-weighted sequences included in  list:', len(sequence_t2w))
    return patient_id, sequence_t2w, ijk_t2w, zone, zone_label


def buildNumpyArray(patient_id, sequence_id, path=r'Data'):
    lstFilesDCM = []
    slices = []
    PathDicom = path +'\\' + patient_id + '\\'+ sequence_id

    print(PathDicom)

    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():
                lstFilesDCM.append(os.path.join(dirName,filename))
    
  
    skipcount = 0
    for file in lstFilesDCM:
        if hasattr(pydicom.read_file(file), 'SliceLocation'):
            slices.append(pydicom.read_file(file))
        else:
            skipcount = skipcount + 1
        
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    
    #Read reference dicom for header specifications
    RefDs = slices[0]

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    for ds in slices:
        # store the raw image data
        ArrayDicom[:, :, slices.index(ds)] = ds.pixel_array

    return ArrayDicom

def dicomLoader(patient_id, sequence_id, path=r'Data'):
    # List for storing patient sequences
    multi_sequence = []

#    # Loop through all patients
    for i in range(len(patient_id)):
        patient_numpy_array = buildNumpyArray(patient_id[i], sequence_id[i], path='data')
        multi_sequence.append(patient_numpy_array)
    return multi_sequence
