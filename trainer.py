import argparse 
import pickle
from tqdm import tqdm

from tweet import Tweet

def main():

    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--test_data', required=True)

    ARGS = parser.parse_args()

    with open(ARGS.train_data, 'rb') as handle:
        train_data = pickle.load(handle)

    with open(ARGS.test_data, 'rb') as handle:
        test_data = pickle.load(handle)

    feature_vectors = []

    for tweet in train_data:
        tweet_feature_vector = []
        #insert feature extraction functions
        


if __name__ == "__main__":
    main()