import pickle 
import argparse
import matplotlib.pyplot as plt

def main():

    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_counts', required=True)
    parser.add_argument('--real_counts', required=True)

    ARGS = parser.parse_args()

    with open(ARGS.model_counts, 'rb') as handle:
        model_counts = pickle.load(handle)

    with open(ARGS.real_counts, 'rb') as handle:
        real_counts = pickle.load(handle)
    

if __name__ == "__main__":
    main()