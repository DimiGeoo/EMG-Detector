import pandas as pd
import matplotlib.pyplot as plt


def visualize_all_gestures(data, replies):
    data = data[~data["target"].isin(["paramesos", "peace"])]
    classes = data["target"].unique()

    plt.figure(figsize=(12, 6))

    # Iterate through each class and plot its data
    for i, class_label in enumerate(classes, start=1):
        class_data = data[data["target"] == class_label]["feature"].values[
            28600 : 28600 + 1860 * replies
        ]  # Adjust slice as per dataset
        plt.plot(
            range(len(class_data)),
            class_data,
            label=f"Class: {class_label}",
            color=f"C{i}",
        )

    plt.title("Gestures for All Classes")
    plt.legend()
    plt.grid(True)
    plt.ylim(450, 900)  # Adjust based on your dataset's value range
    plt.tight_layout()
    plt.show()


# Plot all gestures on a single plot
def visualize_all_gaestures(data, replies):
    classes = data["target"].unique()

    plt.figure(figsize=(12, 6))

    for i, class_label in enumerate(classes, start=1):
        class_data = data[data["target"] == class_label]["feature"].values[
            28600 : 28600 + 1860 * replies
        ]
        filtered_data = class_data
        plt.plot(
            range(len(filtered_data)),
            filtered_data,
            label=f"Class: {class_label}",
            color=f"C{i}",
        )

    plt.title("Gestures for all classes")
    plt.legend()
    plt.grid(True)
    plt.ylim(450, 900)  # Adjust y-axis range if needed
    plt.tight_layout()
    plt.show()


# Main function
def main():
    file_path = (
        r"D:\GitHub\git\Personal\EMG-Detector\MachineLearning\Datasets\dataset.csv"
    )
    data = pd.read_csv(file_path)
    visualize_all_gestures(data, 5)


if __name__ == "__main__":
    main()
