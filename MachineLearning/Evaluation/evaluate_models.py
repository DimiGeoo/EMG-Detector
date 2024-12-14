from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  # SVM Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import joblib
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load EMG data from a CSV file
# data = pd.read_csv("MachineLearning/Datasets/labeled_data.csv")
data = pd.read_csv("MachineLearning/Datasets/labeled_data_1.csv")
excluded_classes = []
data = data[~data["target"].isin(excluded_classes)]


# Step 2: Create a function to process a window of data and extract features
def process_window(window):
    rms = np.sqrt(np.mean(np.square(window)))  # Root Mean Square (RMS)
    mav = np.mean(np.abs(window))  # Mean Absolute Value (MAV)
    zc = np.sum(np.diff(np.sign(window)) != 0)  # Zero Crossing (ZC)
    wl = np.sum(np.abs(np.diff(window)))  # Waveform Length (WL)

    # Frequency feature: FFT dominant frequency
    fft_vals = fft(window)
    fft_power = np.abs(fft_vals[: len(fft_vals) // 2])
    dominant_freq = np.argmax(fft_power)  # Dominant frequency index

    return [rms, mav, zc, wl, dominant_freq]


# Step 3: Define a window size and step size (200 ms with 50% overlap)
sampling_rate = 1000  # in Hz (samples per second)
window_duration_seconds = 0.5  # 500 ms
window_size = int(sampling_rate * window_duration_seconds)  # Convert to samples
step_size = window_size // 2  # 50% overlap

# Step 4: Segment the data into windows and extract features
features = []
labels = []

for start in range(0, len(data), step_size):
    end = start + window_size
    if end > len(data):
        break
    window = data["feature"].iloc[start:end].values
    features.append(process_window(window))
    labels.append(data["target"].iloc[start])

# Step 5: Convert the features into a DataFrame
features_df = pd.DataFrame(
    features, columns=["RMS", "MAV", "ZC", "WL", "Dominant_Freq"]
)
features_df["target"] = labels

# Step 6: Split the data into features (X) and target (y)
X = features_df.drop("target", axis=1)
y = features_df["target"]

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)


# Step 8: Train and evaluate multiple models
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return model


# Initialize and evaluate models
models = {
    "LDA": LinearDiscriminantAnalysis(),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    ),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Bagging Classifier": BaggingClassifier(n_estimators=100, random_state=42),
    "Ridge Classifier": RidgeClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

trained_models = {}

print("\n--- Model Evaluation Results ---")
for model_name, model in models.items():
    trained_models[model_name] = evaluate_model(model, model_name)

# Step 9: Save all trained models
print("\n--- Saving Trained Models ---")
for model_name, model in trained_models.items():
    filename = f"MachineLearning/Binary_models/{model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, filename)
    print(f"{model_name} saved as {filename}")

print("\nAll models trained, evaluated, and saved successfully!")
