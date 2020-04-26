import csv 
import argparse
import os 
import pickle
from tqdm import tqdm

from tweet import Tweet

def main():
    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', required=True)
    parser.add_argument('--output_directory', required=True)

    ARGS = parser.parse_args()

    all_tweets = []

    for file_index in tqdm(range(len(os.listdir(ARGS.data_directory)))):
        file = os.listdir(ARGS.data_directory)[file_index]
        if file.startswith("2020"):
            with open(ARGS.data_directory + "/" + file) as f:
                csv_reader = csv.reader(f, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                        continue #ignore first line -- just template info

                    line_count += 1
                    datetime = row[2]
                    date = datetime.split("T")[0]
                    time = datetime.split("T")[1].strip("Z")
                    text = row[4]
                    fav_count = row[11]
                    retweet_count = row[12]
                    country_code = row[13] #just country
                    place_full_name = row[14] #city, country
                    place_type = row[15]
                    follower_count = row[16]
                    lang = row[21]
                    
                    single_tweet = Tweet(date, time, text, fav_count, retweet_count, country_code, place_full_name, follower_count)
                    all_tweets.append(single_tweet)

    #perform split
    TRAIN_END_INDEX = int(len(all_tweets)*0.8)

    train_tweets = all_tweets[:TRAIN_END_INDEX]
    test_tweets = all_tweets[TRAIN_END_INDEX:]
    small_tweets = all_tweets[:100]

    with open(ARGS.output_directory+"/raw_tweets.train.pickle", 'wb') as handle:
        pickle.dump(train_tweets, handle)

    with open(ARGS.output_directory+"/raw_tweets.test.pickle", 'wb') as handle:
        pickle.dump(test_tweets, handle)

    with open(ARGS.output_directory+"/raw_tweets.small.pickle", 'wb') as handle:
        pickle.dump(small_tweets, handle)

if __name__ == "__main__":
    main()