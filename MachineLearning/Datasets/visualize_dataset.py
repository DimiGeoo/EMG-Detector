""" Visualize each class of CSV into a plot. """

import csv
import matplotlib.pyplot as plt
from collections import defaultdict

# CSV_FILE = "MachineLearning/Datasets/labeled_data.csv"
CSV_FILE = "MachineLearning/Datasets/labeled_data_1.csv"


# Read the CSV and group values by label
def read_and_group_data(csv_file):
    label_data = defaultdict(list)
    try:
        with open(csv_file, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = row["target"]
                value = float(row["feature"])
                label_data[label].append(value)
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
    return label_data


# Plot all classes
def visualize_gestures_with_subplots(label_data):
    labels = list(label_data.keys())
    num_gestures = len(labels)

    plt.figure(figsize=(12, 8))  # Set window size big enough for the classes

    for i, (label, values) in enumerate(label_data.items(), start=1):
        plt.subplot(num_gestures, 1, i)  # Create subplots in a column
        plt.plot(range(len(values)), values, label=label, color="C" + str(i))
        plt.title(f"Gesture: {label}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    label_data = read_and_group_data(CSV_FILE)
    if label_data:
        visualize_gestures_with_subplots(label_data)
    else:
        print("No data to visualize.")


if __name__ == "__main__":
    main()
