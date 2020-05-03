#! /usr/bin/env bash

python trainer.py \
    --all_data data/covid/raw_tweets.test.pickle \
    --weights pretrained_weights/simple_mlp.h5 \
    --output_counts output_counts.pickle \