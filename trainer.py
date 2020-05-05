import argparse 
import pickle
from tqdm import tqdm
from joblib import dump, load

import tensorflow as tf
import numpy as np 
from sklearn import svm
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from langdetect import detect

from tweet import Tweet
from models import *
from covid_features import *

def extract_features(data):
    """
    Helper function to call each of the feature extraction functions
    """

    #load in pretrained model
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    all_feature_vectors = []
    dates = []

    for tweet_index in tqdm(range(len(data))):
        tweet = data[tweet_index]
        tweet_feature_vector = []
        if tweet.country_code == "US":
            content = tweet.content
            try:
                if detect(content) == 'en':
                    #calling feature extraction functions
                    #------------------------------------
                    #tweet_feature_vector.append(determine_length(content))
                    tweet_feature_vector.append(count_infection_words(content))
                    tweet_feature_vector.append(count_possession_words(content))
                    tweet_feature_vector.append(count_concern_words(content))
                    tweet_feature_vector.append(count_vaccination_words(content))
                    tweet_feature_vector.append(count_symptom_words(content))
                    tweet_feature_vector.append(count_cdc_words(content))
                    #tweet_feature_vector.append(count_positive_emoticons(content))
                    tweet_feature_vector.append(count_negative_emoticons(content))
                    tweet_feature_vector.append(count_mentions(content))
                    tweet_feature_vector.append(count_hashtags(content))
                    #tweet_feature_vector.append(contains_url(content))
                    #------------------------------------
                    
                    tweet_embedding = embed([content]) #512 dimension vector
                    assert tweet_embedding.shape[1] == 512

                    tweet_embedding = np.ravel(tweet_embedding)
                    tweet_feature_vector = np.asarray(tweet_feature_vector)
                    tweet_feature_vector = np.concatenate((tweet_embedding, tweet_feature_vector), axis=0)

                    all_feature_vectors.append(tweet_feature_vector)

                    #collect date for future validation
                    if tweet.date:
                        tweet_date = tweet.date.split("-")
                        tweet_date = tweet_date[1] + "/" + tweet_date[2] + "/20"
                        dates.append(tweet_date)
            except:
                continue
    return np.asarray(all_feature_vectors), np.asarray(dates)

def main():

    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_data', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output_counts', required=True)
    parser.add_argument('--output_predictions', required=True)

    ARGS = parser.parse_args()

    with open(ARGS.all_data, 'rb') as handle:
        all_data = pickle.load(handle)

    all_feature_vectors, dates = extract_features(all_data)
    
    if "joblib" in ARGS.weights:
        classifier = load(ARGS.weights) 
        predictions = classifier.predict(all_feature_vectors)
    elif "h5" in ARGS.weights:
        model = load_model(ARGS.weights)
        predictions = model.predict(all_feature_vectors)
    else:
        raise Exception("Pretrained weights file format not supported yet")

    #evaluation metrics

    date_counts = {}
    for prediction, date in zip(predictions, dates):
        if prediction != '0':
            if date in date_counts:
                date_counts[date] += 1
            else:
                date_counts[date] = 1

    covid_tweets = []
    for tweet, prediction in zip(all_data, predictions):
        if prediction == 1:
            covid_tweets.append(tweet.content)

    with open(ARGS.output_counts, 'wb') as handle:
        pickle.dump(date_counts, handle)

    with open(ARGS.output_predictions, 'wb') as handle:
        pickle.dump(covid_tweets, handle)
        
if __name__ == "__main__":
    main()