#Author: Gentry Atkinson
#Organization: Texas University
#Data: 7 April, 2021
#Visualize the segmentation of a signal with 3 methods

#Method 1: regular breaks every 150 samples
#Method 2: 150 samples centered on a PIP
#Method 3: Divide segments at PIPs, resample each segment to 150

from scipy.signal import resample
from fastpip import pip
from matplotlib import pyplot as plt
import pandas
import random

if __name__ == "__main__":
    instances = pandas.read_csv('pip_test_data.csv')
    print('Keys: ',  instances.keys())
    print('D types: ', instances.dtypes)
    print('Number of instances: ', len(instances['samples']))

    lucky_winner = random.randint(0,5000)

    s = instances['samples'][lucky_winner]
    s = s[1:-2]
    s = s.replace('\n', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    s = [float(x) for x in s.split(' ') if x]

    #Plot the segment with divisions every 150 samples

    reg_divisions = list(range(150, len(s), 150))

    plt.plot(range(len(s)), s)
    for x in reg_divisions:
        plt.axvline(x=x, color='red')
    plt.savefig('plot_marked_every_150.pdf')

    #Plot the segment with PIPs shown as red lines

    plt.figure()

    NUM_SEGMENTS = len(s)//150
    pips = pip([(i, j) for i,j in enumerate(s)], NUM_SEGMENTS+1) #n+1 pips create n segments
    trim_pips = pip([(i, j) for i,j in enumerate(s)], NUM_SEGMENTS+2)
    trim_pips = trim_pips[1:-1] #remove start and end points

    plt.plot(range(len(s)), s)
    for (x, y) in trim_pips:
        plt.axvline(x=x, color='red')
    plt.savefig('plot_marked_with_pips.pdf')

    #Plot segments produced by 3 methods

    plt.figure()
    plt.plot(range(150,300, 1), s[150:300])
    plt.savefig('one_seg_by_m1.pdf')

    plt.figure()
    plt.plot(range(150), s[trim_pips[1][0]-75:trim_pips[1][0]+75])
    plt.savefig('one_seg_by_m2.pdf')

    plt.figure()
    plt.plot(range(150), resample(s[pips[1][0]:pips[2][0]], 150))
    plt.savefig('one_seg_by_m3.pdf')


    #plt.show()
