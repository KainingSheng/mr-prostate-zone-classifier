# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:51:14 2019

@author: KainingSheng
"""
import numpy as np
import cv2
from sklearn.preprocessing import normalize

def getLesionCoordinates(ijk_t2w):
    x_t2w = [];
    y_t2w = [];
    z_t2w = [];
    for t2w_ijk in ijk_t2w:
        t2w_ijk = t2w_ijk.split(' ')
        x_t2w.append(int(t2w_ijk[0]))
        y_t2w.append(int(t2w_ijk[1]))
        z_t2w.append(int(t2w_ijk[2]))
    return x_t2w, y_t2w, z_t2w
    
        

def cropLesion(sequenceArrayList, X, Y, Z, plus=15, minus=15):
    sequenceArrayList_resized = []
    for array,x,y,z in zip(sequenceArrayList,X,Y,Z):
        (h, w, d) = np.shape(array)
        w_center = x
        h_center = y
        d_center = z
    
        h_start = int(h_center - minus)
        h_end = int(h_center + plus)
        w_start = int(w_center - minus)
        w_end = int(w_center + plus)

        crop = array[h_start:h_end, w_start:w_end, d_center]
        sequenceArrayList_resized.append(crop)
    return sequenceArrayList_resized


def resize2D (sequenceArray, w =96, h =96):
    sequenceResized = []
    for array in sequenceArray:
        array =cv2.resize(array, (h, w), interpolation=cv2.INTER_CUBIC)
        sequenceResized.append(array)

    return sequenceResized


def addSequence2D3(t2w_cropped, adc_resized, dce_resized):
    addedSequence = []
    for t2w, adc,dce in zip(t2w_cropped, adc_resized,dce_resized):
        t2w = np.expand_dims(t2w,axis=2)
        adc = np.expand_dims(adc,axis=2)
        dce = np.expand_dims(dce,axis=2)
        concatSequence = np.concatenate((t2w,adc,dce),axis=2)
        addedSequence.append(concatSequence)
    return addedSequence

def normalizer(sequenceArray):
    sequence_normalized = []
    for sequence in sequenceArray:
        sequence_norm = normalize(sequence)
        sequence_normalized.append(sequence_norm)
    return sequence_normalized
