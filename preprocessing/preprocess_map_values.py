import csv 
import argparse
import pickle
from tqdm import tqdm

def main():
    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--output_file', required=True)

    ARGS = parser.parse_args()
    values = {}
    dates = []

    with open(ARGS.data_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for date in row[11:]:
                    dates.append(date)
                line_count += 1
                continue #ignore first line -- just template info
            province_state = row[6]
            if province_state not in values:
                values[province_state] = {}
            for count_index in range(len(row[11:])):
                count = row[11+count_index]
                values[province_state][dates[count_index]] = int(count)
                
            line_count += 1

    with open(ARGS.output_file, 'wb') as handle:
        pickle.dump(values, handle)

if __name__ == "__main__":
    main()