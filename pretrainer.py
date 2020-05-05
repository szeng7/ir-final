#!/usr/bin/env python3
import argparse 
import pickle
from tqdm import tqdm
from joblib import dump, load

import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from tensorflow.keras import regularizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from sklearn.linear_model import LogisticRegression

from tweet import Tweet
from models import *
from flu_features import *

RANDOM_SEED = 21

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

def extract_features(data):
    """
    Helper function to call each of the feature extraction functions
    """

    #load in pretrained model
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    all_feature_vectors = []

    for content_index in tqdm(range(len(data))):
    
        content = data[content_index]

        tweet_feature_vector = []
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

    return np.asarray(all_feature_vectors).astype('float32')


def main():

    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--model_output_file', required=False)
    parser.add_argument('--model_architecture', required=True)
    parser.add_argument('--optimizer', required=False)
    parser.add_argument('--learning_rate', required=False, type=float)
    parser.add_argument('--loss', required=False)
    parser.add_argument('--num_epochs', required=False, type=int)
    parser.add_argument('--batch_size', required=False, type=int)

    ARGS = parser.parse_args()

    with open(ARGS.train_data, 'rb') as handle:
        train_x, train_y = pickle.load(handle)

    with open(ARGS.test_data, 'rb') as handle:
        test_x, test_y = pickle.load(handle)

    #select the right model

    architectures = ['simple_mlp', 'mlp']
    function_architecture_mapping = {
        'simple_mlp': simple_mlp,
        'mlp': mlp,
    }

    train_feature_vectors = extract_features(train_x)
    test_feature_vectors = extract_features(test_x)


    if ARGS.model_architecture == "svc":
        classifier = SVC(kernel="rbf", gamma="auto")
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
        predictions = classifier.predict(test_feature_vectors)

        #evaluation metrics
        total = 0
        correct = 0
        for pred_y, true_y in zip(predictions, test_y):
            if pred_y == true_y:
                correct += 1
            total += 1

        print(f"Test Set Accuracy: {correct / total:.2f}")

        if ARGS.model_output_file:
            print(f"Saved model to {ARGS.model_output_file}")
            dump(classifier, ARGS.model_output_file)

    elif ARGS.model_architecture == "logreg":
        classifier = LogisticRegression().fit(train_feature_vectors, train_y)
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
        predictions = classifier.predict(test_feature_vectors)

        #evaluation metrics
        total = 0
        correct = 0
        for pred_y, true_y in zip(predictions, test_y):
            if pred_y == true_y:
                correct += 1
            total += 1

        print(f"Test Set Accuracy: {correct / total:.2f}")

        if ARGS.model_output_file:
            print(f"Saved model to {ARGS.model_output_file}")
            dump(classifier, ARGS.model_output_file)

    elif ARGS.model_architecture == "svm":
        #train
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
        predictions = classifier.predict(test_feature_vectors)

        #evaluation metrics
        total = 0
        correct = 0
        for pred_y, true_y in zip(predictions, test_y):
            if pred_y == true_y:
                correct += 1
            total += 1

        print(f"Test Set Accuracy: {correct / total:.2f}")

        if ARGS.model_output_file:
            print(f"Saved model to {ARGS.model_output_file}")
            dump(classifier, ARGS.model_output_file)

    elif ARGS.model_architecture not in architectures:
        
        raise Exception("Model chosen has not been implemented")
    
    else:
        
        model_name = function_architecture_mapping[ARGS.model_architecture]
        model = model_name(train_feature_vectors.shape[1])
        np_train_y = np.asarray(train_y).astype('float32')

        if ARGS.loss == "binary_crossentropy":
            loss = tf.keras.losses.BinaryCrossentropy()
        else:
            raise Exception("Other loss functions not supported yet")

        if ARGS.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=ARGS.learning_rate)
        else:
            raise Exception("Other optimizers not supported yet")

        checkpoint = ModelCheckpoint(ARGS.model_output_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1)

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(x=train_feature_vectors, y=np_train_y, batch_size=ARGS.batch_size, epochs=ARGS.num_epochs, shuffle=True, validation_data =[test_feature_vectors, np.asarray(test_y).astype('float32')], callbacks = [checkpoint])
        
        #model.save_weights(ARGS.model_output_file)

if __name__ == "__main__":
    main()
