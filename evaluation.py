import pickle 
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from scipy.stats import pearsonr

def main():

    #parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_counts', required=True)
    parser.add_argument('--real_counts', required=True)
    parser.add_argument('--output_predictions', required=True)

    ARGS = parser.parse_args()

    with open(ARGS.model_counts, 'rb') as handle:
        model_counts = pickle.load(handle)

    with open(ARGS.real_counts, 'rb') as handle:
        real_counts = pickle.load(handle)

    with open(ARGS.output_predictions, 'rb') as handle:
        output_predictions = pickle.load(handle)

    print(ARGS.model_counts)
    predicted_dates = []
    predicted_counts = []
    for key, value in sorted(model_counts.items()):
        predicted_dates.append(key)
        predicted_counts.append(value)

    true_dates = []
    true_counts = []
    for key, value in real_counts.items():
        true_dates.append(key)
        true_counts.append(value)

    plt.plot(predicted_dates, predicted_counts)
    plt.title("Predicted COVID-19 Counts from Trained SVM")
    plt.ylabel("Case Count")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("model_counts.png")
    plt.clf()
    plt.cla()

    x_ticks = []

    for label_index in range(len(true_dates)):
        if label_index % 14 == 0:
            x_ticks.append(true_dates[label_index])

    plt.plot(true_dates, true_counts)
    plt.axvline(sorted(predicted_dates)[0].lstrip("0"), 0, 999999999, color='#008000', linestyle="--", label='model output start')
    plt.legend()
    plt.ylabel("Case Count")
    plt.xlabel("Date")
    plt.title("True COVID-19 Counts from JHU CSSE")
    plt.tight_layout()
    plt.xticks(x_ticks)
    plt.savefig("actual_counts.png")

    #calculate pearson coefficient
    modified_real_counts = []

    remove_zero = []
    for date in predicted_dates:
        newdate = date.split("/")[0].lstrip("0") + "/" + date.split("/")[1].lstrip("0") + "/20"
        remove_zero.append(newdate)
    predicted_dates = remove_zero

    for date in predicted_dates:
        modified_real_counts.append(real_counts[date])

    print(f"Pearson Correlation: {pearsonr(predicted_counts, modified_real_counts)}")
    

if __name__ == "__main__":
    main()