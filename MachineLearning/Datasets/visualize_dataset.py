import csv
import matplotlib.pyplot as plt
from collections import defaultdict

# File containing labeled data
CSV_FILE = "MachineLearning/Datasets/labeled_data.csv"


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


# Plot the labeled data
def visualize_data(label_data):
    plt.figure(figsize=(10, 6))
    for label, values in label_data.items():
        plt.plot(
            range(len(values)), values, label=label
        )  # X-axis: index, Y-axis: values

    plt.title("Labeled Data Visualization")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend(title="Labels")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    label_data = read_and_group_data(CSV_FILE)
    if label_data:
        visualize_data(label_data)
    else:
        print("No data to visualize.")


if __name__ == "__main__":
    main()
