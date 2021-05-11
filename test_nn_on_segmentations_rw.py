#Author: Gentry Atkinson
#Organization: Texas University
#Data: 11 May, 2021
#Train and test a NN on 3 segmentation methods.

#Method 1: regular breaks every 150 samples
#Method 2: 150 samples centered on a PIP
#Method 3: Divide segments at PIPs, resample each segment to 150

import utils.build_simple_dnn as bs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from utils.ts_feature_toolkit import get_features_from_one_signal
from tensorflow.keras.utils import to_categorical

#Open results file
results_file = open('results/NN_test_results_rw.txt', 'w+')

#Read file for segmentation method 1
all_m1 = np.genfromtxt('seg_by_reg_division_rw.csv', delimiter=',')
m1_samples = [i[1:] for i in all_m1]
m1_labels = [i[0] for i in all_m1]
print('M1 Number of instances: ', len(m1_samples))
print('M1 Number of labels: ', len(m1_labels))
print('M1 Length of instances: ', len(m1_samples[0]))
m1_features = [get_features_from_one_signal(i) for i in m1_samples]
print('M1 Number of feature vectors: ', len(m1_features))
print('M1 Feature vector length: ', len(m1_features[0]))

#Fit NN to method 1 segments w/o feature_extraction
results_file.write('NN on Method 1, No Feature extraction\n')
X_train, X_test, y_train, y_test = map(np.array, train_test_split(m1_samples, m1_labels, shuffle=True, test_size=0.2))
y_train = to_categorical(y_train)
test_NN = bs.get_trained_dnn(X_train, y_train)
y_pred = np.argmax(test_NN.predict(X_test), axis=-1)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

#Fit NN to method 1 segments w/ feature_extraction
results_file.write('\n\nNN on Method 1, With Feature extraction\n')
X_train, X_test, y_train, y_test = map(np.array,train_test_split(m1_features, m1_labels, shuffle=True, test_size=0.2))
y_train = to_categorical(y_train)
test_NN = bs.get_trained_dnn(X_train, y_train)
y_pred = np.argmax(test_NN.predict(X_test), axis=-1)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

#Read file for segmentation method 2
all_m2 = np.genfromtxt('seg_cen_on_pip_rw.csv', delimiter=',')
m2_samples = [i[1:] for i in all_m2]
m2_labels = [i[0] for i in all_m2]
print('M2 Number of instances: ', len(m2_samples))
print('M2 Number of labels: ', len(m2_labels))
print('M2 Length of instances: ', len(m2_samples[0]))
m2_features = [get_features_from_one_signal(i) for i in m2_samples]
print('M2 Number of feature vectors: ', len(m2_features))
print('M2 Feature vector length: ', len(m2_features[0]))

#Fit NN to method 2 segments w/o feature_extraction
results_file.write('\n\nNN on Method 2, No Feature extraction\n')
X_train, X_test, y_train, y_test =  map(np.array,train_test_split(m2_samples, m2_labels, shuffle=True, test_size=0.2))
y_train = to_categorical(y_train)
test_NN = bs.get_trained_dnn(X_train, y_train)
y_pred = np.argmax(test_NN.predict(X_test), axis=-1)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

#Fit NN to method 2 segments w/ feature_extraction
results_file.write('\n\nNN on Method 2, With Feature extraction\n')
X_train, X_test, y_train, y_test =  map(np.array,train_test_split(m2_features, m2_labels, shuffle=True, test_size=0.2))
y_train = to_categorical(y_train)
test_NN = bs.get_trained_dnn(X_train, y_train)
y_pred = np.argmax(test_NN.predict(X_test), axis=-1)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

#Read file for segmentation method 3
all_m3 = np.genfromtxt('seg_and_resamp_from_pips_rw.csv', delimiter=',')
m3_samples = [i[1:] for i in all_m3]
m3_labels = [i[0] for i in all_m3]
print('M3 Number of instances: ', len(m3_samples))
print('M3 Number of labels: ', len(m3_labels))
print('M3 Length of instances: ', len(m3_samples[0]))
m3_features = [get_features_from_one_signal(i) for i in m3_samples]
print('M3 Number of feature vectors: ', len(m3_features))
print('M3 Feature vector length: ', len(m3_features[0]))

#Fit NN to method 3 segments w/o feature_extraction
results_file.write('\n\nNN on Method 3, No Feature extraction\n')
X_train, X_test, y_train, y_test =  map(np.array,train_test_split(m3_samples, m3_labels, shuffle=True, test_size=0.2))
y_train = to_categorical(y_train)
test_NN = bs.get_trained_dnn(X_train, y_train)
y_pred = np.argmax(test_NN.predict(X_test), axis=-1)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

#Fit NN to method 3 segments w/ feature_extraction
results_file.write('\n\nNN on Method 3, With Feature extraction\n')
X_train, X_test, y_train, y_test =  map(np.array,train_test_split(m3_features, m3_labels, shuffle=True, test_size=0.2))
y_train = to_categorical(y_train)
test_NN = bs.get_trained_dnn(X_train, y_train)
y_pred = np.argmax(test_NN.predict(X_test), axis=-1)
results_file.write('Accuracy: {}\n'.format(accuracy_score(y_test, y_pred, normalize=True)))
results_file.write('Precision: {}\n'.format(precision_score(y_test, y_pred, average='micro')))
results_file.write('Recall: {}\n'.format(recall_score(y_test, y_pred, average='micro')))

results_file.close()
