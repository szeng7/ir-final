#! /usr/bin/env bash

python3 pretrainer.py \
    --train_data data/influenza/influenza.train \
    --test_data data/influenza/influenza.test \
    --model_output_file pretrained_weights/simple_mlp.h5 \
    --model_architecture "simple_mlp" \
    --optimizer "adam" \
    --learning_rate 0.001 \
    --loss "binary_crossentropy" \
    --num_epochs 10 \