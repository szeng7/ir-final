#! /usr/bin/env bash

python trainer.py \
    --all_data data/covid/raw_tweets.all.pickle \
    --weights pretrained_weights/svm_bow.joblib \
    --output_counts svm_bow_output_counts.pickle \
    --bag_of_words "True" \