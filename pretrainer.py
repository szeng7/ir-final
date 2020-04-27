#!/usr/bin/env python3
import argparse 
import pickle
from tqdm import tqdm

import tensorflow as tf
import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split

from tweet import Tweet


def load_tweets(file):
    """Load Tweets from input file.

    Loads Tweets from input file where each line is a tag and Tweet organized
    as the following format:
        <positive/neutral/negative>\t<Tweet content>
    Returns two lists, the first being the Tweets and the second being the labels 
    """

    with open(file, 'r') as fi:
        tweets = []
        labels = []
        for line in fi:
            label, tweet_content = line.split('\t', 1)
            labels.append(label)
            tweet = Tweet
            tweet.content = tweet_content
            tweets.append(tweet)

        return tweets, labels


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

    for tweet in data:
        tweet_feature_vector = []
        content = tweet.content
        #calling feature extraction functions
        #------------------------------------
        tweet_feature_vector.append(determine_length(content))
        tweet_feature_vector.append(0)
        #------------------------------------
        
        all_feature_vectors.append(tweet_feature_vector)

    return np.asarray(all_feature_vectors)


def main():

    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)

    ARGS = parser.parse_args()

    tweets, labels = load_tweets(ARGS.data)

    #split data
    tweets_train, tweets_test, labels_train, labels_test = train_test_split(
            tweets, labels, test_size=0.2, random_state=0)

    #train
    train_feature_vectors = extract_features(tweets_train)
    classifier = svm.SVC()
    classifier.fit(train_feature_vectors, labels_train)
    
    #test
    test_feature_vectors = extract_features(tweets_test)
    predictions = classifier.predict(test_feature_vectors)

    #evaluation metrics

    total = 0
    correct = 0
    for pred_y, true_y in zip(predictions, labels_test):
        if pred_y == true_y:
            correct += 1
        total += 1

    print(f"Test Set Accuracy: {correct / total:.2f}")

if __name__ == "__main__":
    main()
