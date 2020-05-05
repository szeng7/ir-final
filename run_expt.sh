#! /usr/bin/env bash

python trainer.py \
    --all_data data/covid/raw_tweets.test.pickle \
    --weights pretrained_weights/svm_stem.joblib \
    --output_counts new_output_counts.pickle \
    --output_predictions new_output_prediction.pickle \