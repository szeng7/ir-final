import argparse 
import pickle
from tqdm import tqdm

import tensorflow as tf
import numpy as np 
from sklearn import svm

from tweet import Tweet

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
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--global_counts', required=True)
    parser.add_argument('--us_counts', required=True)

    ARGS = parser.parse_args()

    with open(ARGS.train_data, 'rb') as handle:
        train_data = pickle.load(handle)

    with open(ARGS.test_data, 'rb') as handle:
        test_data = pickle.load(handle)

    with open(ARGS.global_counts, 'rb') as handle:
        global_counts = pickle.load(handle)

    with open(ARGS.us_counts, 'rb') as handle:
        us_counts = pickle.load(handle)

    train_feature_vectors = extract_features(train_data)
    
    #hardcoding some fake labels until we get our real dataset
    temp_label = []
    for i in range(50):
        temp_label.append([0])
        temp_label.append([1])
    temp_label = np.asarray(temp_label)

    #model information, will hotswap this with some neural nets later
    classifier = svm.SVC()
    classifier.fit(train_feature_vectors, temp_label.ravel())
    
    test_feature_vectors = extract_features(test_data)
    predictions = classifier.predict(test_feature_vectors)

    #evaluation metrics

    total = 0
    correct = 0
    for pred_y, true_y in zip(predictions, temp_label):
        if pred_y == true_y:
            correct += 1
        total += 1

    print(f"Test Set Accuracy: {correct / total:.2f}")

if __name__ == "__main__":
    main()