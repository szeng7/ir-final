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
import dnn_model

def extract_bow_vector(data, vocab):
    """
    Helper function to create BOW vectors for training
    """
    all_feature_vectors = []
    dates = []

    for tweet_index in tqdm(range(len(data))):
        tweet = data[tweet_index]
        tweet_feature_vector = [0] * len(vocab)
        if tweet.country_code == "US":
            content = tweet.content
            try:
                if detect(content) == 'en':
                    #use vocab to create bow vector
                    content = content.strip("\n")
                    content = re.sub(r'[^\w\s]','', content)
                
                    for word in content.split(" "):
                        word = word.lower()
                        if word in vocab:
                            tweet_feature_vector[vocab[word]] = 1
                        else:
                            tweet_feature_vector[vocab["OOV"]] = 1

                    #collect date for future validation
                    if tweet.date:
                        tweet_date = tweet.date.split("-")
                        tweet_date = tweet_date[1].lstrip("0") + "/" + tweet_date[2] + "/20"
                        dates.append(tweet_date)

                    assert len(tweet_feature_vector) == len(vocab)
                    all_feature_vectors.append(tweet_feature_vector)

            except:
                continue

    return np.asarray(all_feature_vectors), np.asarray(dates)

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
                        tweet_date = tweet_date[1].lstrip("0") + "/" + tweet_date[2] + "/20"
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
    parser.add_argument('--output_predictions', required=False)
    parser.add_argument('--bag_of_words', required=False)

    ARGS = parser.parse_args()

    with open(ARGS.all_data, 'rb') as handle:
        all_data = pickle.load(handle)

    dnn_flag = False
    print(ARGS.weights)
    if ARGS.bag_of_words:
        print("BAG OF WORDS")
        with open("bow_vocab.pickle", 'rb') as handle:
            vocab = pickle.load(handle)
            all_feature_vectors, dates = extract_bow_vector(all_data, vocab)

    else:
        print("FEATURE EXTRACTION")
        all_feature_vectors, dates = extract_features(all_data)

    if "joblib" in ARGS.weights:
        classifier = load(ARGS.weights) 
        predictions = classifier.predict(all_feature_vectors)
    elif "h5" in ARGS.weights:
        model = load_model(ARGS.weights)
        predictions = model.predict(all_feature_vectors)
    elif "pt" in ARGS.weights:
        model = dnn(all_feature_vectors.shape[1])
        predictions = dnn_model.predict_no_eval(model, all_feature_vectors)
        dnn_flag = True

    else:
        raise Exception("Pretrained weights file format not supported yet")

    #evaluation metrics
    print("Starting to collect counts")
    date_counts = {}
    for prediction, date in tqdm(zip(predictions, dates), total=len(dates)):
        if dnn_flag == True:
            prediction = prediction.item() 
        if prediction != '0':
            if date in date_counts:
                date_counts[date] += 1
            else:
                date_counts[date] = 1

    """
    covid_tweets = []
    for tweet, prediction in zip(all_data, predictions):
        if prediction == 1:
            covid_tweets.append(tweet.content)
    """
    print("Writing to pickle...")
    with open(ARGS.output_counts, 'wb') as handle:
        pickle.dump(date_counts, handle)
        print(f"Outputed to {ARGS.output_counts}")

    #with open(ARGS.output_predictions, 'wb') as handle:
    #    pickle.dump(covid_tweets, handle)
        
if __name__ == "__main__":
    main()