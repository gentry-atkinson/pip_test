# PIP Test

Gentry Atkinson | Texas State Universitty | 6 April, 2021

## Puppose

This project will explore the uses of Perceuptually Important Points in a
method for generalized time series embedding.

## Files:
  Create Dataset-> Creates 5000 instances of binary data of varying lengths\
  Create Segmentations-> Creates fixed length instances from raw data by 3 methods\
    **The first value in the seg CSVs is the label of the instance**\
  Initial Visualization-> Plots 2 figures of raw data\
  Segmentation Visuals-> Plots 5 visuals to better understand segmentation.
  Test SVM on Segmentations-> Train and test an SVM on the instance sets created
    by the 3 methods of segmentation.\
  Davies-Bouldin on Segments-> Compute the Davies-Bouldin score for each of the
    3 methods.\
  Test NN on Segmentations-> Repeat the SVM experiment with a neural network.\
