import numpy as np 
from sklearn.model_selection import train_test_split
import pickle

RANDOM_SEED = 21

# unrelated/news (0) vs related (1)

unrelated_vs_related_data = {}
inconsistent_tweet_ids = set()

with open("../data/raw/influenza/RelatedVsNotRelated2009TweetIDs.content.txt", "r", encoding='utf-8') as f:
    for line in f:
        line = line.split("\t")
        tweet_id = int(line[0])
        label = int(line[1])
        content = line[2]

        if tweet_id in unrelated_vs_related_data:
            if label == unrelated_vs_related_data[tweet_id][1]:
                continue
            else:
                # Some tweet id's with inconsistent labels - keep track and delete them afterward
                inconsistent_tweet_ids.add(tweet_id)

        else:
            unrelated_vs_related_data[tweet_id] = (content, label)

with open("../data/raw/influenza/RelatedVsNotRelated2012TweetIDs.content.txt", "r", encoding='utf-8') as f:
    for line in f:
        line = line.split("\t")
        tweet_id = int(line[0])
        label = int(line[1])
        content = line[2]

        if tweet_id in unrelated_vs_related_data:
            if label == unrelated_vs_related_data[tweet_id][1]:
                continue
            else:
                inconsistent_tweet_ids.add(tweet_id)
        else:
            unrelated_vs_related_data[tweet_id] = (content, label)

for tweet_id in inconsistent_tweet_ids:
    del unrelated_vs_related_data[tweet_id]


# infection (0) vs awareness (1)

infection_vs_awareness_data = {}
inconsistent_tweet_ids = set()

with open("../data/raw/influenza/AwarenessVsInfection2009TweetIDs.content.txt", "r", encoding='utf-8') as f:
    for line in f:
        line = line.split("\t")
        tweet_id = int(line[0])
        label = int(line[1])
        content = line[2]

        if tweet_id in infection_vs_awareness_data:
            if label == infection_vs_awareness_data[tweet_id][1]:
                continue
            else:
                # Some tweet id's with inconsistent labels - keep track and delete them afterward
                inconsistent_tweet_ids.add(tweet_id)

        else:
            infection_vs_awareness_data[tweet_id] = (content, label)

with open("../data/raw/influenza/AwarenessVsInfection2012TweetIDs.content.txt", "r", encoding='utf-8') as f:
    for line in f:
        line = line.split("\t")
        tweet_id = int(line[0])
        label = int(line[1])
        content = line[2]

        if tweet_id in infection_vs_awareness_data:
            if label == infection_vs_awareness_data[tweet_id][1]:
                continue
            else:
                # Some tweet id's with inconsistent labels - keep track and delete them afterward
                inconsistent_tweet_ids.add(tweet_id)

        else:
            infection_vs_awareness_data[tweet_id] = (content, label)

for tweet_id in inconsistent_tweet_ids:
    del infection_vs_awareness_data[tweet_id]


# combine data sets - 1 for infected with flu, 0 otherwise
# if there only exists data for infection vs awareness without unrelated/news vs related, it becomes ambiguous
for tweet_id in unrelated_vs_related_data.keys():
    if unrelated_vs_related_data[tweet_id][1] == 1: # if flu related
        # flu related but not infected should become 0
        if tweet_id in infection_vs_awareness_data and infection_vs_awareness_data[tweet_id][1] == 0:
            continue
        else:
            unrelated_vs_related_data[tweet_id] = (unrelated_vs_related_data[tweet_id][0], 0)

x = []
y = []
for key, value in unrelated_vs_related_data.items():
    content, label = value
    x.append(content)
    y.append(label)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)


with open("../data/influenza/influenza.train", 'wb') as handle:
    pickle.dump((x_train, y_train), handle)

with open("../data/influenza/influenza.test", 'wb') as handle:
    pickle.dump((x_test, y_test), handle)
