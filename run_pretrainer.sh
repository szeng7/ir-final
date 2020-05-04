#! /usr/bin/env bash

python3 pretrainer.py \
    --train_data data/influenza/influenza.train \
    --test_data data/influenza/influenza.test \
    --model_output_file pretrained_weights/svm.joblib \
    --model_architecture "svm" \
