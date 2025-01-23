import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Path to your folder containing the CSV files
folder_path = "./MachineLearning/Datasets/New"


# Plot gestures for each class
def visualize_gestures(data):
    classes = data["target"].unique()  # Get unique class labels
    num_classes = len(classes)

    plt.figure(
        figsize=(12, num_classes * 4)
    )  # Adjust height based on the number of classes

    for i, class_label in enumerate(classes, start=1):
        class_data = data[data["target"] == class_label]["feature"].values
        filtered_data = class_data
        plt.subplot(num_classes, 1, i)
        plt.plot(
            range(len(filtered_data)),
            filtered_data,
            label=f"Class: {class_label}",
            color=f"C{i}",
        )
        plt.title(f"Gesture: {class_label}")
        plt.xlabel("Index")
        plt.ylabel("Filtered Value")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Main function
def main():
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    # Load and combine all CSV files
    df_list = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

    # Combine all DataFrames into one
    if df_list:
        data = pd.concat(df_list, ignore_index=True)

        # Ensure the "feature" column is numeric
        data["feature"] = pd.to_numeric(data["feature"], errors="coerce")
        data.dropna(subset=["feature"], inplace=True)

        # Visualize gestures
        visualize_gestures(data)
    else:
        print("No CSV files found in the specified folder.")


if __name__ == "__main__":
    main()
