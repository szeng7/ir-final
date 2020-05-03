import argparse 
import pickle
from tqdm import tqdm
from joblib import dump, load

import tensorflow as tf
import numpy as np 
from sklearn import svm

from tweet import Tweet

from models import *

"""
PUT ALL FEATURE EXTRACTION FUNCTIONS HERE
"""

def determine_length(tweet_content):
    if len(tweet_content) > 10:
        return 1
    else:
        return 0

def extract_features(data):
    """
    Helper function to call each of the feature extraction functions
    """

    all_feature_vectors = []
    dates = []

    for tweet in data:
        tweet_feature_vector = []
        if tweet.country_code == "US":
            content = tweet.content
            #calling feature extraction functions
            #------------------------------------
            tweet_feature_vector.append(determine_length(content))
            tweet_feature_vector.append(0)
            #------------------------------------
            
            all_feature_vectors.append(tweet_feature_vector)

            #collect date for future validation
            if tweet.date:
                tweet_date = tweet.date.split("-")
                tweet_date = tweet_date[1] + "/" + tweet_date[2] + "/20"
                dates.append(tweet_date)

    return np.asarray(all_feature_vectors), np.asarray(dates)

def main():

    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_data', required=True)
    parser.add_argument('--global_counts', required=True)
    parser.add_argument('--us_counts', required=True)
    parser.add_argument('--weights', required=True)

    ARGS = parser.parse_args()

    with open(ARGS.all_data, 'rb') as handle:
        all_data = pickle.load(handle)

    with open(ARGS.global_counts, 'rb') as handle:
        global_counts = pickle.load(handle)

    with open(ARGS.us_counts, 'rb') as handle:
        us_counts = pickle.load(handle)

    all_feature_vectors, dates = extract_features(all_data)
    
    #model information, will hotswap this with some neural nets later
    classifier = load(ARGS.weights) 
    predictions = classifier.predict(all_feature_vectors)

    #evaluation metrics

    predictions = np.append(predictions, "1")
    dates = np.append(dates, "03/03/20")

    date_counts = {}
    for prediction, date in zip(predictions, dates):
        if prediction != '0':
            if date in date_counts:
                date_counts[date] += 1
            else:
                date_counts[date] = 1

    total_us_counts_daily = {}

    for state, date_dict in us_counts.items():
        for date, count in date_dict.items():
            if date not in total_us_counts_daily:
                total_us_counts_daily[date] = count
            else:
                total_us_counts_daily[date] += count

    #add graphing here once we get better data
    
if __name__ == "__main__":
    main()