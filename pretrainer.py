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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from tweet import Tweet
from models import *
import dnn_model
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

def run_scikit_classifier(classifier, train_x, train_y, test_x, test_y, output_file=None):
    """
    Fits specified scikit learn model on training data
    and prints results of model after classifying testing data

    Saves model to file if output_file is not None
    """

    classifier.fit(train_x, train_y)

    #get train accuracy
    predictions = classifier.predict(train_x)
    total = 0
    correct = 0
    for pred_y, true_y in zip(predictions, train_y):
        if pred_y == true_y:
            correct += 1
        total += 1

    print(f"Train Set Accuracy: {correct / total:.2f}")

    #test
    predictions = classifier.predict(test_x)

    #evaluation metrics
    total = 0
    correct = 0
    for pred_y, true_y in zip(predictions, test_y):
        if pred_y == true_y:
            correct += 1
        total += 1

    print(f"Test Set Accuracy: {correct / total:.2f}")

    if output_file:
        print(f"Saved model to {output_file}")
        dump(classifier, output_file)

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
    parser.add_argument('--validation_split', required=False, type=bool)

    ARGS = parser.parse_args()

    # get feature vectors
    with open(ARGS.train_data, 'rb') as handle:
        train_x, train_y = pickle.load(handle)
    train_feature_vectors = extract_features(train_x)


    if ARGS.validation_split:
         train_feature_vectors, test_feature_vectors, train_y, test_y = \
                train_test_split(train_feature_vectors, train_y, test_size=0.2)
    else:
        with open(ARGS.test_data, 'rb') as handle:
            test_x, test_y = pickle.load(handle)
        test_feature_vectors = extract_features(test_x)


    #select the right model
    scikit_models = {
        'svc': SVC(kernel="rbf", gamma="auto"),
        'logreg': LogisticRegression(),
        'svm': svm.SVC(),
        'random_forest': RandomForestClassifier(max_depth=7, n_estimators=500),
        'decision_tree': DecisionTreeClassifier(max_depth=7, max_features='auto'),
        'naive_bays': GaussianNB()
    }

    function_architecture_mapping = {
        'simple_mlp': simple_mlp,
        'mlp': mlp,
        'dnn': dnn
    }

    print('Selected model: ' + ARGS.model_architecture)
    if ARGS.model_architecture in scikit_models:
        run_scikit_classifier(
            scikit_models[ARGS.model_architecture], 
            train_feature_vectors, train_y, 
            test_feature_vectors, test_y,
            output_file=ARGS.model_output_file
        )
    elif ARGS.model_architecture == 'dnn':
        model = dnn(train_feature_vectors.shape[1])
        model = dnn_model.fit(model, train_feature_vectors, train_y)

        acc, predictions = dnn_model.predict(model, train_feature_vectors, train_y)
        print(f"Train Set Accuracy: {acc:.2f}")

        acc, predictions = dnn_model.predict(model, test_feature_vectors, test_y)
        print(f"Test Set Accuracy: {acc:.2f}")
        
        if ARGS.model_output_file:
            print(f"Saved model to {ARGS.model_output_file}")
            dnn_model.save_model(model, ARGS.model_output_file)
    elif ARGS.model_architecture in function_architecture_mapping:
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
    else:
        raise Exception("Model chosen has not been implemented")

if __name__ == "__main__":
    main()
