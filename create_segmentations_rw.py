#Author: Gentry Atkinson
#Organization: Texas University
#Data:11 May, 2021
#Create segmentations of the raw data file using 3 methds

#Method 1: regular breaks every 150 samples
#Method 2: 150 samples centered on a PIP
#Method 3: Divide segments at PIPs, resample each segment to 150

from scipy.signal import resample
from fastpip import pip
import numpy as np
import pandas
import os

files = [
    #'dog1_walk1.csv',
    #'dog1_walk2.csv',
    'human1_walk1.csv',
    'human1_walk2.csv'
]

if __name__ == "__main__":
    os.system('rm seg_by_reg_division_rw.csv')
    os.system('rm seg_cen_on_pip_rw.csv')
    os.system('rm seg_and_resamp_from_pips_rw.csv')
    m1_file = open('seg_by_reg_division_rw.csv', 'a+')
    m2_file = open('seg_cen_on_pip_rw.csv', 'a+')
    m3_file = open('seg_and_resamp_from_pips_rw.csv', 'a+')
    for file in files:
        print("Reading file: ", file)
        instances = pandas.read_csv('raw_data/'+file)
        print('Keys: ',  instances.keys())
        print('D types: ', instances.dtypes)
        print('Number of samples: ', len(instances['time']))

        instance = instances['ax']
        print('Still number of sample: ', len(instance))



        l = '0' if 'dog' in file else '1'
        print('Label of this instance: ', l)

        #Write segmentations using method 1
        for j in range(0, len(instance)-150, 150):
            m1_file.write(l + ', ' + ', '.join([str(x) for x in instance[j:j+150]]) + '\n')
            #print(l)

        num_segments = len(instance)//150
        pips = pip([(i, j) for i,j in enumerate(instance)], num_segments+1, distance='vertical') #n+1 pips create n segments
        trim_pips = pip([(i, j) for i,j in enumerate(instance)], num_segments+2, distance='vertical')
        trim_pips = trim_pips[1:-1] #remove start and end points

        #Write segmentations using method 2
        for p in trim_pips:
            if p[0]<75:
                m2_file.write(l + ', ' + ', '.join(str(i) for i in instance[:150]) + '\n')
            elif len(instance) - p[0] < 76:
                m2_file.write(l + ', ' + ', '.join(str(i) for i in instance[-150:]) + '\n')
            else:
                m2_file.write(l + ', ' + ', '.join(str(i) for i in instance[p[0]-75:p[0]+75]) + '\n')
            #print(l)

        #Write segmentations using method 3
        for j in range(len(pips)-1):
            m3_file.write(l + ', ' + ', '.join([str(t) for t in resample(instance[pips[j][0]:pips[j+1][0]], 150)]) + '\n')
            #print(l)

    print('done')


    m1_file.close()
    m2_file.close()
    m3_file.close()
