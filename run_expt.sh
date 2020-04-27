#! /usr/bin/env bash

python trainer.py \
    --train_data data/raw_tweets.small.pickle \
    --test_data data/raw_tweets.small.pickle \
    --global_counts data/global_count.pickle \
    --us_counts data/us_count.pickle \