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
    print('Preparing 5000 instances of 500 to 1500 seconds (10 to 30 seconds at 50Hz)')
    classes = [
        {
            'avg_pattern_length':randint(3, 7),
            'avg_amplitude':randint(1, 3),
            'default_variance':randint(1, 3),
            'variance_pattern_length': randint(5, 20),
            'variance_amplitude': randint(1, 5)
        },
        {
            'avg_pattern_length':randint(3, 7),
            'avg_amplitude':randint(1, 3),
            'default_variance':randint(1, 3),
            'variance_pattern_length': randint(5, 20),
            'variance_amplitude': randint(1, 5)
        },
    ]
    print('Preparing data with ', len(classes), ' classes.')
    instances = []
    for i in range(2500):
        instances.append(
            {
                'instance number': i+1,
                'samples': generate_pattern_data_as_array(
                    length=randint(500, 1500),
                    avg_pattern_length=classes[0]['avg_pattern_length'],
                    default_variance=classes[0]['default_variance'],
                    variance_pattern_length=classes[0]['variance_pattern_length'],
                    variance_amplitude=classes[0]['variance_amplitude']
                )
            }
        )
    for i in range(2500):
        instances.append(
            {
                'instance number': 2500+i+1,
                'samples': generate_pattern_data_as_array(
                    length=randint(500, 1500),
                    avg_pattern_length=classes[1]['avg_pattern_length'],
                    default_variance=classes[1]['default_variance'],
                    variance_pattern_length=classes[1]['variance_pattern_length'],
                    variance_amplitude=classes[1]['variance_amplitude']
                )
            }
        )
    instance_frame = pandas.DataFrame(instances)
    #savemat('pip_test_data.mat', instances)
    #print(instance_frame)
    print('Keys: ', instances[0].keys())
    print('Shortest instance: ', min([len(i['samples']) for i in instances]))
    print('Longest instance: ', max([len(i['samples']) for i in instances]))
    instance_frame.to_csv('pip_test_data.csv')
    print('done')
