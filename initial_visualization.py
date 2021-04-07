#Author: Gentry Atkinson
#Organization: Texas University
#Data: 6 April, 2021
#Make some line graphs of our test data

import pandas
from matplotlib import pyplot as plt
from utils.ts_feature_toolkit import get_features_from_one_signal
import numpy as np
from sklearn.manifold import TSNE as tsne

if __name__ == "__main__":
    instances = pandas.read_csv('pip_test_data.csv')
    print('Keys: ',  instances.keys())
    print('D types: ', instances.dtypes)
    print('Number of instances: ', len(instances['samples']))

    plt.figure(1)
    plt.title("4 Instances of Each Class")
    plt.subplot(211)
    for s in instances['samples'][:4]:
        s = s[1:-2]
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = [float(x) for x in s.split(' ') if x]
        # print(len(s))
        plt.plot(s, c='blue')
    plt.subplot(212)
    for s in instances['samples'][2500:2504]:
        s = s[1:-2]
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = [float(x) for x in s.split(' ') if x]
        # print(len(s))
        plt.plot(s, c='red')
    plt.savefig('pip_test_data_raw_plot.pdf')
    #plt.show()

    features = []
    for s in instances['samples'][:]:
        s = s[1:-2]
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = [float(x) for x in s.split(' ') if x]
        features.append(get_features_from_one_signal(np.array(s), 50))

    plt.figure(2)
    vis = tsne(n_components=2, n_jobs=8).fit_transform(features)
    for v in vis[0:2500]:
        plt.scatter(v[0], v[1], c='blue', marker=".")
    for v in vis[2500:5000]:
        plt.scatter(v[0], v[1], c='red', marker=".")
    plt.savefig('pip_test_data_tsne.pdf')
