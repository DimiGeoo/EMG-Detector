import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
file_path = r"D:\GitHub\git\Personal\EMG-Detector\MachineLearning\Datasets\processed_features_dataset.csv"
data = pd.read_csv(file_path)

# Filter out any unwanted classes if needed
data = data[~data["target"].isin(["paramesos", "peace"])]

# Define a consistent color palette for all classes
unique_classes = data["target"].unique()
class_colors = sns.color_palette("Set2", len(unique_classes))
palette = dict(zip(unique_classes, class_colors))

# Create box plots with consistent colors
plt.figure(figsize=(12, 6))
sns.boxplot(x="target", y="RMS", data=data, palette=palette)
plt.title("Boxplot of RMS by Target Class")
plt.xlabel("Target Class")
plt.ylabel("RMS Value")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="target", y="MAV", data=data, palette=palette)
plt.title("Boxplot of MAV by Target Class")
plt.xlabel("Target Class")
plt.ylabel("MAV Value")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="target", y="WL", data=data, palette=palette)
plt.title("Boxplot of WL by Target Class")
plt.xlabel("Target Class")
plt.ylabel("WL Value")
plt.show()
