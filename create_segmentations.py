#Author: Gentry Atkinson
#Organization: Texas University
#Data: 6 April, 2021
#Create segmentations of the raw data file using 3 methds

#Method 1: regular breaks every 150 samples
#Method 2: 150 samples centered on a PIP
#Method 3: Divide segments at PIPs, resample each segment to 150

from scipy.signal import resample
from fastpip import pip
import pandas

if __name__ == "__main__":
    instances = pandas.read_csv('pip_test_data.csv')
    print('Keys: ',  instances.keys())
    print('D types: ', instances.dtypes)
    print('Number of instances: ', len(instances['samples']))

    m1_file = open('seg_by_reg_division.csv', 'w+')
    m2_file = open('seg_cen_on_pip.csv', 'w+')
    m3_file = open('seg_and_resamp_from_pips.csv', 'w+')

    for (i,s) in enumerate(instances['samples']):
        s = s[1:-2]
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = [float(x) for x in s.split(' ') if x]

        #Write segmentations using method 1
        for i in range(0, len(s)-150, 150):
            m1_file.write(' '.join([str(x) for x in s[i:i+150]]) + '\n')

        num_segments = len(s)//150
        pips = pip([(i, j) for i,j in enumerate(s)], num_segments+1) #n+1 pips create n segments
        trim_pips = pip([(i, j) for i,j in enumerate(s)], num_segments+2)
        trim_pips = trim_pips[1:-1] #remove start and end points

        #Write segmentations using method 2
        for p in trim_pips:
            if p[0]<75:
                m2_file.write(' '.join(str(i) for i in s[:150]) + '\n')
            elif len(s) - p[0] < 75:
                m2_file.write(' '.join(str(i) for i in s[-150:]) + '\n')
            else:
                m2_file.write(' '.join(str(i) for i in s[p[0]-74:p[0]+75]) + '\n')

        #Write segmentations using method 3
        for i in range(len(pips)-1):
            m3_file.write(' '.join([str(t) for t in resample(s[pips[i][0]:pips[i+1][0]], 150)]) + '\n')

    print('done')


    m1_file.close()
    m2_file.close()
    m3_file.close()
