#! /usr/bin/env bash

#python combine_tweets.py

#python preprocess_tweets.py --data_directory ../coronavirus-covid19-tweets --output_directory ../data

python preprocess_map_values.py --data_file ../data/raw/counts/time_series_covid19_confirmed_US.csv --output_file ../data/counts/us_count.pickle
python preprocess_map_values.py --data_file ../data/raw/counts/time_series_covid19_confirmed_global.csv --output_file ../data/counts/global_count.pickle
