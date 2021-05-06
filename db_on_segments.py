#Author: Gentry Atkinson
#Organization: Texas University
#Data: 5 May, 2021
#Compute the DB Index for each of the 3 segments

#Method 1: regular breaks every 150 samples
#Method 2: 150 samples centered on a PIP
#Method 3: Divide segments at PIPs, resample each segment to 150

import numpy as np
from sklearn.metrics import davies_bouldin_score

#Open the results file
results = open('results/db_index_results.txt', 'w+')
results.write('DB Index for each segmentation method.\n')
results.write('Lower values indicate more dissimilar clusters.\n\n')

#Open the data files
all_m1 = np.genfromtxt('seg_by_reg_division.csv', delimiter=',')
m1_samples = [i[1:] for i in all_m1]
m1_labels = [i[0] for i in all_m1]
print('M1 Number of instances: ', len(m1_samples))
print('M1 Number of labels: ', len(m1_labels))
print('M1 Length of instances: ', len(m1_samples[0]))

all_m2 = np.genfromtxt('seg_cen_on_pip.csv', delimiter=',')
m2_samples = [i[1:] for i in all_m2]
m2_labels = [i[0] for i in all_m2]
print('M2 Number of instances: ', len(m2_samples))
print('M2 Number of labels: ', len(m2_labels))
print('M2 Length of instances: ', len(m2_samples[0]))

all_m3 = np.genfromtxt('seg_and_resamp_from_pips.csv', delimiter=',')
m3_samples = [i[1:] for i in all_m3]
m3_labels = [i[0] for i in all_m3]
print('M3 Number of instances: ', len(m3_samples))
print('M3 Number of labels: ', len(m3_labels))
print('M3 Length of instances: ', len(m3_samples[0]))

results.write('DB Index for regular segmentation (Method 1): {}\n'.format(davies_bouldin_score(m1_samples, m1_labels)))
results.write('DB Index for segments centered on PIPs (Method 2): {}\n'.format(davies_bouldin_score(m2_samples, m2_labels)))
results.write('DB Index for segments between PIPs (Method 3): {}\n'.format(davies_bouldin_score(m3_samples, m3_labels)))
