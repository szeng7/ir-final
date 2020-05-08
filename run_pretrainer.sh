#! /usr/bin/env bash

python3 pretrainer.py \
    --train_data data/influenza/influenza.train \
    --test_data data/influenza/influenza.test \
    --model_output_file pretrained_weights/dnn_stem.joblib \
    --model_architecture "dnn" \
    --optimizer "adam" \
    --learning_rate 0.001 \
    --loss "binary_crossentropy" \
    --num_epochs 100 \
    --batch_size 32 \
    #--bag_of_words "True" \