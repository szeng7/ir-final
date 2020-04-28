import numpy as np 
from sklearn.model_selection import train_test_split
import pickle

RANDOM_SEED = 21

all_data = {}

with open("../data/raw/influenza/RelatedVsNotRelated2009TweetIDs.content.txt", "r") as f:
    for line in f:
        line = line.split("\t")
        tweet_id = line[0]
        label = line[1]
        content = line[2]

        if tweet_id in all_data:
            if label == all_data[tweet_id][1]:
                continue
            else:
                del all_data[tweet_id]
        else:
            all_data[tweet_id] = (content, label)

with open("../data/raw/influenza/RelatedVsNotRelated2012TweetIDs.content.txt", "r") as f:
    for line in f:
        line = line.split("\t")
        tweet_id = line[0]
        label = line[1]
        content = line[2]

        if tweet_id in all_data:
            if label == all_data[tweet_id][1]:
                continue
            else:
                del all_data[tweet_id]
        else:
            all_data[tweet_id] = (content, label)

with open("../data/raw/influenza/AwarenessVsInfection2009TweetIDs.content.txt", "r") as f:
    for line in f:
        line = line.split("\t")
        tweet_id = line[0]
        label = line[1]
        content = line[2]

        if tweet_id in all_data:
            if all_data[tweet_id][1] == 0:
                continue
            elif label == 0:
                all_data[tweet_id] = (content, 1) #infection AND influenza related
        else:
            if label == 0: #switch labels around, 1 should be infection, 0 is awareness
                all_data[tweet_id] = (content, 1)
            else:
                all_data[tweet_id] = (content, 0)

with open("../data/raw/influenza/AwarenessVsInfection2012TweetIDs.content.txt", "r") as f:
    for line in f:
        line = line.split("\t")
        tweet_id = line[0]
        label = line[1]
        content = line[2]

        if tweet_id in all_data:
            continue
        else:
            if label == 0: #switch labels around, 1 should be infection, 0 is awareness
                all_data[tweet_id] = (content, 1)
            else:
                all_data[tweet_id] = (content, 0)
x = []
y = []
for key, value in all_data.items():
    content, label = value
    x.append(content)
    y.append(label)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)

with open("../data/influenza/influenza.train", 'wb') as handle:
        pickle.dump((x_train, y_train), handle)

with open("../data/influenza/influenza.test", 'wb') as handle:
        pickle.dump((x_test, y_test), handle)
