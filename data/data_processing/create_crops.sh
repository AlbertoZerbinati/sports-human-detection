#!/bin/bash

# Alberto Zerbinati

# download yolov3 weights, names, and cfg
wget https://pjreddie.com/media/files/yolov3.weights -O data/data_processing/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O data/data_processing/coco.names
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O data/data_processing/yolov3.cfg

# create directories if they don't exist
mkdir data/positive
mkdir data/negative
mkdir data/dataset

# remove all previous crops
rm -rf data/positive/*
rm -rf data/negative/*
rm -rf data/dataset/*

# extract positive crops
python3 data/data_processing/extract_people_from_dataset.py  \
    --data_folder data/images \
    --save_folder data/positive \
    --keep_percentage 1 \

# extract negative crops
python3 data/data_processing/extract_background_from_dataset.py \
    --data_folder data/images \
    --save_folder data/negative \
    --keep_percentage 1 \
    --num_crops 7

# split into train, valid, test sets
./data/data_processing/split_train_valid_test.sh

# remove positive and negative folders
rm -rf data/positive
rm -rf data/negative
