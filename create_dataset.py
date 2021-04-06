#Author: Gentry Atkinson
#Organization: Texas University
#Data: 6 April, 2021
#Create a synthetic dataset of variable length time series data

from random import randint
from utils.gen_ts_data import generate_pattern_data_as_array
#from scipy.io import savemat
import pandas


'''
This script will produce 5000 instances of random-ish, variable length data.
The two classes are generated randomly. The output will be saved as a .csv.

Maybe I'll normalize the data before writing it.
'''

if __name__ == "__main__":
    print('Preparing 5000 instances of 500 to 1000 seconds (10 to 20 seconds at 50Hz)')
    classes = [
        {
            'avg_pattern_length':3,
            'avg_amplitude':1,
            'default_variance':1,
            'variance_pattern_length': 10,
            'variance_amplitude': 2
        },
        {
            'avg_pattern_length':7,
            'avg_amplitude':3,
            'default_variance':3,
            'variance_pattern_length': 15,
            'variance_amplitude': 4
        },
    ]
    print('Preparing data with ', len(classes), ' classes.')
    instances = []
    for i in range(2500):
        instances.append(
            {
                'instance_number': i+1,
                'samples': generate_pattern_data_as_array(
                    length=randint(500, 1000),
                    avg_pattern_length=classes[0]['avg_pattern_length'],
                    default_variance=classes[0]['default_variance'],
                    variance_pattern_length=classes[0]['variance_pattern_length'],
                    variance_amplitude=classes[0]['variance_amplitude'],
                ),
                'label':int(0)
            }
        )
    for i in range(2500):
        instances.append(
            {
                'instance_number': 2500+i+1,
                'samples': generate_pattern_data_as_array(
                    length=randint(500, 1000),
                    avg_pattern_length=classes[1]['avg_pattern_length'],
                    default_variance=classes[1]['default_variance'],
                    variance_pattern_length=classes[1]['variance_pattern_length'],
                    variance_amplitude=classes[1]['variance_amplitude']
                ),
                'label':int(1)
            }
        )
    instance_frame = pandas.DataFrame(instances)
    #savemat('pip_test_data.mat', instances)
    #print(instance_frame)
    print('Keys: ', instances[0].keys())
    print('Shortest instance: ', min([len(i['samples']) for i in instances]))
    print('Longest instance: ', max([len(i['samples']) for i in instances]))
    instance_frame.to_csv('pip_test_data.csv', columns=['instance_number', 'samples', 'label'], sep=',', index=False)
    print('done')
