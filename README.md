# ir-final

### Instructions

`run_preprocessing.sh` is a script that runs all the preprocessing (should be unnecessary unless recreating from scratch/raw data since I've included necessary pickles in `data`). It runs `preprocess_data.py`, which expects that there's a directory in the repo called `coronavirus-covid19-tweets`, which can be generated by opening up the zip from the kaggle https://www.kaggle.com/smid80/coronavirus-covid19-tweets. directory. I only pushed the small pickle in there so that we could work with some data when creating the pipeline. Note that these pickles contains lists of "Tweet" objects, which are defined in `tweet.py`

Then this script runs `preprocess_map_values.py` which uses `data/time_series_covid19_confirmed_global.csv` and `data/time_series_covid19_confirmed_US.csv` to create two new pickles `US_count.pickle` and `global_count.pickle` which both contains time series case counts for the US's states and each country in each respective file. Note that the format of the structure in the pickle is a dictionary (with country name or state as the key and with its value being another dictionary, which maps a string date (M/DD/YY) to the integer count).