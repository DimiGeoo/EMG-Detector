from sklearn.svm import SVC  # Import the SVM classifier
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load your captured EMG data from a CSV file
data = pd.read_csv("MachineLearning/Datasets/labeled_data.csv")


# Step 2: Create a function to process a window of data and extract time-domain features
def process_window(window):
    rms = np.sqrt(np.mean(np.square(window)))  # Root Mean Square (RMS)
    mav = np.mean(np.abs(window))  # Mean Absolute Value (MAV)
    zc = np.sum(np.diff(np.sign(window)) != 0)  # Zero Crossing (ZC)
    return [rms, mav, zc]


# Step 3: Define a window size and step size (e.g., 200 ms window with 100 ms overlap)
window_size = 50  # in ms (250 samples)
step_size = 25  # in ms (125 samples)
sampling_rate = 5000  # in Hz (samples per second)

# Convert window size and step size to number of samples
window_samples = int(window_size * (sampling_rate / 1000))  # Convert ms to samples
step_samples = int(step_size * (sampling_rate / 1000))  # Convert ms to samples

# Step 4: Segment the data into windows and extract features
features = []
labels = []

for start in range(0, len(data), step_samples):
    end = start + window_samples
    if end > len(data):
        break
    window = data["feature"].iloc[start:end].values
    features.append(process_window(window))
    labels.append(
        data["target"].iloc[start]
    )  # Assuming target is the same for each window

# Step 5: Convert the features into a DataFrame
features_df = pd.DataFrame(features, columns=["RMS", "MAV", "ZC"])
features_df["target"] = labels

# Step 6: Split the data into features (X) and target (y)
X = features_df[["RMS", "MAV", "ZC"]]
y = features_df["target"]

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Step 8: Train an LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Save the trained model
print("-----------------------------------------")
print("\nLinear Discriminant Analysis Model\n")

# Step 9: Evaluate the model
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train, y_train)


print("-----------------------------------------")

print("-----------------------------------------")
print("\nLinear SVM Model\n")

# Step 9: Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("-----------------------------------------")


# Save the trained model
joblib.dump(svm, "MachineLearning/Binary_models/svm_model.pkl")
print("Models saved as 'MachineLearning/Binary_models/svm_model.pkll'")

joblib.dump(lda, "MachineLearning/Binary_models/lda_model.pkl")
print("Model saved as 'MachineLearning/Binary_models/lda_model.pkl'")
