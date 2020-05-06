# ir-final

### Instructions

#### Preprocessing

There are various scripts that were used to preprocess that data included in `preprocessing`. However, since we have included pickles or txt files of the postprocessed data, these files are included just as guidance as to how we parsed the raw data and are not needed to run experiments if you clone the repo.

As a summary:

`run_preprocessing.sh` is a script that runs all the preprocessing. It first runs `combine_tweets.py`, which opens up the influenza related data in `../data/raw/influenza` to first combine and remove duplicates before fixing all the labels so that "1s" represent only tweets that are influenza related AND informative about infection (rather than awareness).

It then runs `preprocess_data.py`, which expects that there's a directory in the repo called `coronavirus-covid19-tweets`, which can be generated by opening up the zip from the kaggle https://www.kaggle.com/smid80/coronavirus-covid19-tweets. directory. I only pushed the small pickle in there so that we could work with some data when creating the pipeline. Note that these pickles contains lists of "Tweet" objects, which are defined in `tweet.py`

Then this script runs `preprocess_map_values.py` which uses `../data/raw/counts/time_series_covid19_confirmed_global.csv` and `../data/raw/counts/time_series_covid19_confirmed_US.csv` to create two new pickles `../data/counts/US_count.pickle` and `../data/counts/global_count.pickle` which both contains time series case counts for the US's states and each country in each respective file. Note that the format of the structure in the pickle is a dictionary (with country name or state as the key and with its value being another dictionary, which maps a string date (M/DD/YY) to the integer count).

### Pretraining (Influenza)

To run the "pretraining" phase, run `./run_pretrainer`. This will run pretrainer.py.

Note that we used the Google Universal Sentence encoder because of its simplistic nature, using a deep averaging network (DAN), combined with a large amount of supervised and unsupervised data (wikidata, snli corpus, q&a datasets, etc). In doing so, it captures the semantic meaning of entire sentences (rather than individual words, for which aggregation means to create sentence embeddings usually prove problematic) and is able to achieve high accuracies on downstream tasks such as semantic textual similarity, sentiment analysis, classification, etc. **Since this model is rather large, the first time you run this script, it'll take a while to download.** 

### Target Task (COVID-19)

To run the target test phase, run `./run_expt`. This will run trainer.py.

### Results

| Train Accuracy  | Test Accuracy  | Features Used  | Model Used  | 
|---|---|---|---|
|  90 |  84 |  all paper features, google encoder embeddings, symptom word check, no stemming, tweet length |  SVM |
|  88 |  84 |  all paper features, google encoder embeddings, symptom word check, no stemming, tweet length |  Deep Neural Net |
|  90 |  84 |  all paper features, google encoder embeddings, symptom word check, no stemming, count cdc words |  SVM |
|  95 |  83 |  all paper features, google encoder embeddings, symptom word check, no stemming, tweet length |  Random Forest |
|  92 |  83 |  all paper features, google encoder embeddings, symptom word check, stemming, count cdc words |  SVM |
|  88 |  84 |  all paper features, google encoder embeddings, symptom word check, stemming, count cdc words |  logreg |
|  90 |  78 |  all paper features, google encoder embeddings, symptom word check, stemming, count cdc words |  Decision Tree |
|  78 |  79 |  all paper features, google encoder embeddings, symptom word check, stemming, count cdc words |  svm gauss kernel |
|  81 |  76 |  all paper features, google encoder embeddings, symptom word check, stemming, count cdc words | Naive Bayes |
