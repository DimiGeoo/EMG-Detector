# Built-in Python Libraries
import os

# Data Manipulation and Processing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Saving and Loading Models
import joblib

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your folder containing the CSV files
csv_path = "./MachineLearning/Datasets/dataset.csv"
conf_matrix_path = "./MachineLearning/conf_matrix"
models_path = "./MachineLearning/Binary_Models"
dataset_path = "./MachineLearning/Datasets"

# Combine all dataframes into one
data = pd.read_csv(csv_path)
excluded_classes = ["paramesos", "peace"]
data = data[~data["target"].isin(excluded_classes)]


# Feature extraction
def process_window(window):
    rms = np.sqrt(np.mean(np.square(window)))  # Root Mean Square
    mav = np.mean(np.abs(window))  # Mean Absolute Value
    wl = np.sum(np.abs(np.diff(window)))  # Waveform Length

    return [rms, mav, wl]


# Windowing parameters
sampling_rate = 1860  # in Hz
window_duration_seconds = 1.3  # 1 sec
window_size = int(sampling_rate * window_duration_seconds)
step_size = window_size

# Segment the data
features = []
labels = []

for start in range(0, len(data), step_size):
    end = start + window_size
    if end > len(data):
        break
    window = data["feature"].iloc[start:end].values
    features.append(process_window(window))
    labels.append(data["target"].iloc[start])

# Create DataFrame for features and labels
features_df = pd.DataFrame(
    features,
    columns=[
        "RMS",
        "MAV",
        "WL",
    ],
)
features_df["target"] = labels

# Split data
X = features_df.drop("target", axis=1)
y = features_df["target"]
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Initialize models
models = {
    "MLP_10_10": MLPClassifier(
        hidden_layer_sizes=(10, 10), max_iter=500, random_state=42
    ),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel="linear", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM Linear": SVC(kernel="linear", random_state=42),
    "LDA": LinearDiscriminantAnalysis(
        solver="lsqr",
        tol=1e-3,
    ),
    # More algorithms to test
    # "Perceptron": Perceptron(),
    # "Logistic Regression": LogisticRegression(),
    # "AdaBoost": AdaBoostClassifier(),
    # "Bagging Classifier": BaggingClassifier(),
    # "Ridge Classifier": RidgeClassifier(),
    # "LDA": LinearDiscriminantAnalysis(),
}

os.makedirs(conf_matrix_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)


# Perform cross-validation and hyperparameter tuning
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y),
        yticklabels=np.unique(y),
    )
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{conf_matrix_path}/{model_name.replace(' ', '_').lower()}.png")
    plt.close()

    return model


# Train and evaluate each model
trained_models = {}

print("\n--- Model Evaluation Results ---")
for model_name, model in models.items():
    trained_models[model_name] = evaluate_model(model, model_name)

# Save the models
print("\n--- Saving Trained Models ---")
for model_name, model in trained_models.items():
    filename = f"{models_path}/{model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, filename, compress=3)  # Compress to level 3
    print(f"{model_name} saved as {filename}")

joblib.dump(scaler, f"{models_path}/scaler.pkl")
print("\nAll models trained, evaluated, and saved successfully!")
# Save the features after extraction
features_df.to_csv(f"{dataset_path}/dataset_extracted.csv", index=False)
