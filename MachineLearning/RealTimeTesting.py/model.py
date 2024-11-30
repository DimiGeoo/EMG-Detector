import os
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
import joblib
import time
import numpy as np
import pandas as pd
from collections import deque

# Load environment variables from .env file
load_dotenv()

# MQTT connection details from environment variables
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = os.getenv("MQTT_TOPIC", "#")

SVM_MODEL_PATH = "./MachineLearning/Binary_models/lda_model.pkl"
# SVM_MODEL_PATH = ./MachineLearning/Binary_models/svm_model.pkl


WINDOW_SIZE = (
    250  # Number of samples for the window (e.g., 250 samples for 50ms at 5000Hz)
)
SAMPLING_RATE = 5000  # Hz
PREDICTION_INTERVAL = 0.1  # Interval for predictions in seconds

model = joblib.load(SVM_MODEL_PATH)

# A queue to store the latest EMG data samples for processing
data_queue = deque(maxlen=WINDOW_SIZE)

# Time tracking for the prediction interval
last_prediction_time = time.time()


# Feature extraction function
def process_window(window):
    rms = np.sqrt(np.mean(np.square(window)))  # Root Mean Square (RMS)
    mav = np.mean(np.abs(window))  # Mean Absolute Value (MAV)
    zc = np.sum(np.diff(np.sign(window)) != 0)  # Zero Crossing (ZC)
    return [rms, mav, zc]  # Return a list of 3 features


# MQTT callback functions
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code: {rc}")
    client.subscribe(TOPIC)  # Subscribe to the desired topic


def on_message(client, userdata, msg):
    message = msg.payload.decode("utf-8")
    try:
        value = float(message)  # Assuming numeric payload
        if len(data_queue) >= WINDOW_SIZE:
            data_queue.popleft()  # Remove the oldest data point if the queue is full
        data_queue.append(value)
    except ValueError:
        print(f"Invalid data received: {message}")


# Real-time prediction function
def process_data():
    global last_prediction_time
    if len(data_queue) == WINDOW_SIZE:
        # Prepare the data window for prediction
        window_data = np.array(data_queue)  # Raw window data (250 samples)
        features = process_window(window_data)  # Extract 3 features (RMS, MAV, ZC)

        # Convert features to a pandas DataFrame with feature names
        feature_names = ["RMS", "MAV", "ZC"]
        features_df = pd.DataFrame([features], columns=feature_names)

        # Predict using the model
        predicted_label = model.predict(features_df)[0]

        # Print the predicted label every 2 seconds
        current_time = time.time()
        if current_time - last_prediction_time >= PREDICTION_INTERVAL:
            print(f"Predicted label: {predicted_label}")
            last_prediction_time = current_time


# Main function
def main():
    # Setup MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)  # Set MQTT credentials

    # Connect to MQTT Broker
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    print("Starting real-time classification...")

    try:
        while True:
            process_data()  # Check for predictions every loop iteration
            time.sleep(0.1)  # Small sleep to control the loop speed
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
