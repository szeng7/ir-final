# ir-final

### Instructions

#### Preprocessing

There are various scripts that were used to preprocess that data included in `preprocessing`. However, since we have included pickles or txt files of the postprocessed data, these files are included just as guidance as to how we parsed the raw data, however, due to file size restriction on github, we opted to leave them out of our repo.

As a summary:

`run_preprocessing.sh` is a script that runs all the preprocessing. It first runs `combine_tweets.py`, which opens up the influenza related data in `../data/raw/influenza` to first combine and remove duplicates before fixing all the labels so that "1s" represent only tweets that are influenza related AND informative about infection (rather than awareness).

It then runs `preprocess_data.py`, which expects that there's a directory in the repo called `coronavirus-covid19-tweets`, which can be generated by opening up the zip from the kaggle https://www.kaggle.com/smid80/coronavirus-covid19-tweets. directory. I only pushed the small pickle in there so that we could work with some data when creating the pipeline. Note that these pickles contains lists of "Tweet" objects, which are defined in `tweet.py`

Then this script runs `preprocess_map_values.py` which uses `../data/raw/counts/time_series_covid19_confirmed_global.csv` and `../data/raw/counts/time_series_covid19_confirmed_US.csv` to create two new pickles `../data/counts/US_count.pickle` and `../data/counts/global_count.pickle` which both contains time series case counts for the US's states and each country in each respective file. Note that the format of the structure in the pickle is a dictionary (with country name or state as the key and with its value being another dictionary, which maps a string date (M/DD/YY) to the integer count).

### Pretraining (Influenza)

To run the "pretraining" phase, run `./run_pretrainer`. This will run pretrainer.py.


### Target Task (COVID-19)

To run the target test phase, run `./run_expt`. This will run trainer.py.
