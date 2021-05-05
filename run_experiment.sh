#!/usr/bin/env bash

#Author: Gentry Atkinson
#Organization: Texas University
#Data: 05 May, 2021
#This create a new synthetic dataset, segment it by 3 methods, create some visuals,
#and test ML models on the segmented data.

#Method 1: regular breaks every 150 samples
#Method 2: 150 samples centered on a PIP
#Method 3: Divide segments at PIPs, resample each segment to 150

echo Creating Dataset
python3 create_datset.py
echo Segmenting Dataset
python3 create_segmentations.py
echo Creating Dataset Visualizations
python3 initial_visualizations.py
echo Creating Segmentation Visualizations
python3 segmentation_visuals.py
echo Training and Testing SVM
python3 test_SVM_on_segmentations.py
