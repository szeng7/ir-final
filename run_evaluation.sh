#! /usr/bin/env bash

python evaluation.py \
    --model_counts test_svc_stem_output_counts.pickle \
    --real_counts data/counts/us_total_counts.pickle \
    --output_predictions bow_output_prediction.pickle \