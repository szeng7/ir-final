#! /usr/bin/env bash

python evaluation.py \
    --model_counts new_output_counts.pickle \
    --real_counts data/counts/us_total_counts.pickle \
    --output_predictions new_output_prediction.pickle \