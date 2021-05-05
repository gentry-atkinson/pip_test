#Author: Gentry Atkinson
#Organization: Texas University
#Data: 5 May, 2021
#Train and test an SVM on 3 segmentation methods.

#Method 1: regular breaks every 150 samples
#Method 2: 150 samples centered on a PIP
#Method 3: Divide segments at PIPs, resample each segment to 150

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
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
results_file.write('SVM on Method 1, No Feature extraction\n')
X_train, X_test, y_train, y_test = train_test_split(m1_samples, m1_labels, shuffle=True, test_size=0.2)
test_SVM = svm.SVC()
test_SVM.fit(X_train, y_train)
y_pred = test_SVM.predict(X_test)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

#Fit SVM to method 1 segments w/ feature_extraction
results_file.write('\n\nSVM on Method 1, With Feature extraction\n')
X_train, X_test, y_train, y_test = train_test_split(m1_features, m1_labels, shuffle=True, test_size=0.2)
test_SVM = svm.SVC()
test_SVM.fit(X_train, y_train)
y_pred = test_SVM.predict(X_test)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

#Read file for segmentation method 2
all_m2 = np.genfromtxt('seg_cen_on_pip.csv')
m2_samples = [i[1:] for i in all_m2]
m2_labels = [i[0] for i in all_m2]
print('M2 Number of instances: ', len(m2_samples))
print('M2 Number of labels: ', len(m2_labels))
print('M2 Length of instances: ', len(m2_samples[0]))
m2_features = [get_features_from_one_signal(i) for i in m2_samples]
print('M2 Number of feature vectors: ', len(m2_features))
print('M2 Feature vector length: ', len(m2_features[0]))

#Fit SVM to method 2 segments w/o feature_extraction
results_file.write('\n\nSVM on Method 2, No Feature extraction\n')
X_train, X_test, y_train, y_test = train_test_split(m2_samples, m2_labels, shuffle=True, test_size=0.2)
test_SVM = svm.SVC()
test_SVM.fit(X_train, y_train)
y_pred = test_SVM.predict(X_test)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

#Fit SVM to method 2 segments w/ feature_extraction
results_file.write('\n\nSVM on Method 2, With Feature extraction\n')
X_train, X_test, y_train, y_test = train_test_split(m2_features, m2_labels, shuffle=True, test_size=0.2)
test_SVM = svm.SVC()
test_SVM.fit(X_train, y_train)
y_pred = test_SVM.predict(X_test)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

results_file.close()
