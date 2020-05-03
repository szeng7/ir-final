#! /usr/bin/env bash

python trainer.py \
    --all_data data/covid/raw_tweets.test.pickle \
    --global_counts data/counts/global_count.pickle \
    --us_counts data/counts/us_count.pickle \
    --weights pretrained_weights/svm.joblib \