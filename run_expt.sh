#! /usr/bin/env bash

python trainer.py \
    --all_data data/covid/raw_tweets.test.pickle \
    --weights pretrained_weights/svm.joblib \
    --output_counts output_counts.pickle \