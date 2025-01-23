from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
import pandas as pd
import os

# Path to your folder containing the CSV files
folder_path = "./MachineLearning/Datasets/New"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# Initialize an empty list to store dataframes
df_list = []

# Loop through each CSV file, read it, and append to the list
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df_list.append(df)

# Combine all dataframes into one
data = pd.concat(df_list, ignore_index=True)
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
window_duration_seconds = 1  # 1 sec
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
    "LDA": LinearDiscriminantAnalysis(),
    "SVM (Linear)": SVC(kernel="linear", random_state=42),
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

# Create directory for confusion matrices
os.makedirs("MachineLearning/conf_matrix", exist_ok=True)


# Perform cross-validation and hyperparameter tuning
def evaluate_model(model, model_name):
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
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
    plt.savefig(
        f"MachineLearning/conf_matrix/{model_name.replace(' ', '_').lower()}_conf_matrix.png"
    )
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
    filename = f"MachineLearning/Binary_models/{model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, filename, compress=3)  # Compress to level 3
    print(f"{model_name} saved as {filename}")

joblib.dump(scaler, "MachineLearning/Binary_models/scaler.pkl")
print("\nAll models trained, evaluated, and saved successfully!")
features_df.to_csv(
    "MachineLearning/Datasets/121221processed_features_dataset.csv", index=False
)
