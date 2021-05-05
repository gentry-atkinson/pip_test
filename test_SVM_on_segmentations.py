#Author: Gentry Atkinson
#Organization: Texas University
#Data: 5 May, 2021
#Train and test an SVM on 3 segmentation methods.

#Method 1: regular breaks every 150 samples
#Method 2: 150 samples centered on a PIP
#Method 3: Divide segments at PIPs, resample each segment to 150

from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from utils.ts_feature_toolkit import get_features_from_one_signal

#Open results file
results_file = open('SVM_test_results.txt', 'w+')

#Read file for segmentation method 1
all_m1 = np.genfromtxt('seg_by_reg_division.csv')
m1_samples = [i[1:] for i in all_m1]
m1_labels = [i[0] for i in all_m1]
print('M1 Number of instances: ', len(m1_samples))
print('M1 Number of labels: ', len(m1_labels))
print('M1 Length of instances: ', len(m1_samples[0]))
m1_features = [get_features_from_one_signal(i) for i in m1_samples]
print('M1 Number of feature vectors: ', len(m1_features))
print('M1 Feature vector length: ', len(m1_features[0]))

#Fit SVM to method 1 segments w/o feature_extraction
X_train, X_test, y_train, y_test = train_test_split(m1_samples, m1_labels, shuffle=True, test_size=0.2)
test_SVM = svm.SVC()
test_SVM.fit(m1_samples, m1_labels)
