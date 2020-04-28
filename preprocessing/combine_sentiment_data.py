"""Combines the Tweet sentiment files.

Should not need to be run after file is generated.
"""

def combine(base_path, input_file_names, new_file_name):
    """Combines files and creates combined_sentiment.txt"""

    with open(base_path + new_file_name, 'w+') as output: 
        for file_name in input_file_names:
            with open(base_path + file_name) as file:
                for line in file:
                    output.write(line)


def main():
    base_path = 'data/twitter_sentiment/'
    input_file_names = ['semeval_train.txt', 'Twitter2013_raw.txt',
            'Twitter2014_raw.txt', 'Twitter2015_raw.txt', 'Twitter2016_raw.txt']

    combined_file_name = 'combined_sentiment.txt'

    combine(base_path, input_file_names, combined_file_name)
    

if __name__ == "__main__":
    main()
