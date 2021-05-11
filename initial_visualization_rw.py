#Author: Gentry Atkinson
#Organization: Texas University
#Data: 11 May, 2021
#Make some line graphs of our test data

import numpy as np
from sklearn.manifold import TSNE as tsne
from matplotlib import pyplot as plt
from utils.ts_feature_toolkit import get_features_from_one_signal

m1_file = 'seg_by_reg_division_rw.csv'
m2_file = 'seg_cen_on_pip_rw.csv'
m3_file = 'seg_and_resamp_from_pips_rw.csv'



if __name__ == "__main__":
    m1_segments = np.genfromtxt(m1_file, delimiter=',')
    m2_segments = np.genfromtxt(m2_file, delimiter=',')
    m3_segments = np.genfromtxt(m3_file, delimiter=',')

    m1_features = []
    m1_labels = []
    for s in m1_segments:
        m1_features.append(get_features_from_one_signal(s[1:]))
        m1_labels.append(int(s[0]))
    print('Total number of m1 instances: ', len(m1_features))
    print('Total number of m1 labels: ', len(m1_labels))
    print('Total length of m1 feature vector: ', len(m1_features[0]))

    m2_features = []
    m2_labels = []
    for s in m2_segments:
        m2_features.append(get_features_from_one_signal(s[1:]))
        m2_labels.append(int(s[0]))
    print('Total number of m2 instances: ', len(m2_features))
    print('Total number of m2 labels: ', len(m2_labels))
    print('Total length of m2 feature vector: ', len(m2_features[0]))

    m3_features = []
    m3_labels = []
    for s in m3_segments:
        m3_features.append(get_features_from_one_signal(s[1:]))
        m3_labels.append(int(s[0]))
    print('Total number of m3 instances: ', len(m3_features))
    print('Total number of m3 labels: ', len(m3_labels))
    print('Total length of m3 feature vector: ', len(m3_features[0]))

    m1_vis = tsne(n_components=2, n_jobs=8).fit_transform(m1_features)
    m2_vis = tsne(n_components=2, n_jobs=8).fit_transform(m2_features)
    m3_vis = tsne(n_components=2, n_jobs=8).fit_transform(m3_features)

    print('M1 tsne points: ', len(m1_vis))
    print('M2 tsne points: ', len(m2_vis))
    print('M3 tsne points: ', len(m3_vis))
    print('Method 1 dog instances: ', len(m1_vis[np.where(m1_labels==0.)]))
    print('Method 1 human instances: ', len(m1_vis[np.where(m1_labels==1.)]))

    color = ['red', 'blue']
    plt.figure(1)
    for i in range(len(m1_vis)):
        plt.scatter(m1_vis[i][0], m1_vis[i][1], c=color[m1_labels[i]], marker=".")
    plt.savefig('imgs/pip_rw_m1_data_tsne.pdf')

    plt.figure(2)
    for i in range(len(m2_vis)):
        plt.scatter(m2_vis[i][0], m2_vis[i][1], c=color[m2_labels[i]], marker=".")
    plt.savefig('imgs/pip_rw_m2_data_tsne.pdf')

    plt.figure(3)
    for i in range(len(m3_vis)):
        plt.scatter(m3_vis[i][0], m3_vis[i][1], c=color[m3_labels[i]], marker=".")
    plt.savefig('imgs/pip_rw_m3_data_tsne.pdf')
