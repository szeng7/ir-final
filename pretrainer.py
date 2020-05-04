#!/usr/bin/env python3
import argparse 
import pickle
from tqdm import tqdm

import tensorflow as tf
import numpy as np 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import re

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

infection_words = ['getting', 'got', 'recovered', 'have', 'having', 'had', 'has', 'catching', 'catch', 'cured', 'infected']
possession_words = ['bird', 'the flu', 'flu', 'sick', 'epidemic']
concern_words = ['afraid', 'worried', 'scared', 'fear', 'worry', 'nervous', 'dread', 'dreaded', 'terrified']
vaccination_words = ['vaccine', 'vaccines', 'shot', 'shots', 'mist', 'tamiflu', 'jab', 'nasal spray']
positive_emoticons = [':)', ':D']
negative_emoticons = [':(', ':/']

def count_infection_words(tweet_content):
    count = 0
    for word in infection_words:
        if word in tweet_content:
            count += 1
    return count

def count_possession_words(tweet_content):
    count = 0
    for word in possession_words:
        if word in tweet_content:
            count += 1
    return count

def count_concern_words(tweet_content):
    count = 0
    for word in concern_words:
        if word in tweet_content:
            count += 1
    return count

def count_vaccination_words(tweet_content):
    count = 0
    for word in vaccination_words:
        if word in tweet_content:
            count += 1
    return count

def count_positive_emoticons(tweet_content):
    count = 0
    for word in positive_emoticons:
        if word in tweet_content:
            count += 1
    return count

def count_negative_emoticons(tweet_content):
    count = 0
    for word in negative_emoticons:
        if word in tweet_content:
            count += 1
    return count

def count_mentions(tweet_content):
    return len(re.findall('^@\S+', tweet_content))

def count_hashtags(tweet_content):
    return len(re.findall('^#\S+', tweet_content))

def contains_url(tweet_content):
    return bool(re.search('http[s]?: // (?:[a-zA-Z] |[0-9] |[$-_ @.& +] |[! * \(\),] | (?: %[0-9a-fA-F][0-9a-fA-F]))+', tweet_content))


def extract_features(data):
    """
    Helper function to call each of the feature extraction functions
    """

    all_feature_vectors = []

    for content in data:
        tweet_feature_vector = []
        #calling feature extraction functions
        #------------------------------------
        tweet_feature_vector.append(count_infection_words(content))
        tweet_feature_vector.append(count_possession_words(content))
        tweet_feature_vector.append(count_concern_words(content))
        tweet_feature_vector.append(count_vaccination_words(content))
        tweet_feature_vector.append(count_positive_emoticons(content))
        tweet_feature_vector.append(count_negative_emoticons(content))
        tweet_feature_vector.append(count_mentions(content))
        tweet_feature_vector.append(count_hashtags(content))
        tweet_feature_vector.append(contains_url(content))

        #------------------------------------
        
        all_feature_vectors.append(tweet_feature_vector)

    return np.asarray(all_feature_vectors)


def main():

    #parse input arguments
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--train_data', required=True)
    #parser.add_argument('--test_data', required=True)

    #ARGS = parser.parse_args()

    #with open(ARGS.train_data, 'rb') as handle:
    with open('data/influenza/influenza.train', 'rb') as handle:
        train_x, train_y = pickle.load(handle)

    #with open(ARGS.test_data, 'rb') as handle:
    with open('data/influenza/influenza.test', 'rb') as handle:
        test_x, test_y = pickle.load(handle)

    #vectorizer = TfidfVectorizer()
    #vectorizer = CountVectorizer(ngram_range=(1, 3))
    #docs_tfidf = vectorizer.fit_transform(train_x)

    #train
    train_feature_vectors = extract_features(train_x)
    classifier = svm.SVC()
    classifier.fit(train_feature_vectors, train_y)

    #get train accuracy
    predictions = classifier.predict(train_feature_vectors)
    total = 0
    correct = 0
    for pred_y, true_y in zip(predictions, train_y):
        if pred_y == true_y:
            correct += 1
        total += 1

    print(f"Train Set Accuracy: {correct / total:.2f}")
    
    
    #test
    test_feature_vectors = extract_features(test_x)
    predictions = classifier.predict(test_feature_vectors)

    #evaluation metrics
    total = 0
    correct = 0
    for pred_y, true_y in zip(predictions, test_y):
        if pred_y == true_y:
            correct += 1
        total += 1

    print(f"Test Set Accuracy: {correct / total:.2f}")

if __name__ == "__main__":
    main()
