# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:51:14 2019

@author: KainingSheng
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, normalize
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

from scipy.stats import sem



np.random.seed(1234)
rng=np.random.RandomState(1234)

def plotROC(y_truth, y_pred):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_truth, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()



def getCIAUC(y_truth, y_pred ): 
    n_bootstraps = 2000
    bootstrapped_scores = []   
    
    y_true_int = np.zeros(len(y_truth))
    y_pred_int = np.zeros(len(y_pred))
    for i in range(len(y_truth)):
        n1 = (y_truth[i])
        n2 = (y_pred[i])
        n1 = float(n1)
        n2 = float(n2)
        y_true_int[i] = n1
        y_pred_int[i] = n2
        
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
       
        if len(np.unique(y_true_int[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true_int[indices], y_pred_int[indices])
        bootstrapped_scores.append(score)   
 
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
 
   # 95% c.i.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
   
    plt.hist(bootstrapped_scores, bins=50)
    plt.title('Histogram of the bootstrapped ROC AUC scores')
    plt.show()
    
    print('95% Confidence Interval:[',confidence_lower ,';',confidence_upper,']')
    return confidence_lower,confidence_upper
