from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio, joblib, time, os
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.preprocessing import StandardScaler

# FastAPI App
app = FastAPI()

# Class mapping for classification results
class_mapping = {
    "peace": {
        "name": "peace",
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYcFlwRxQnvJGufTMBqIOhy2CvUiy06p-Mew&s",
    },
    "rest": {
        "name": "rest",
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYcFlwRxQnvJGufTMBqIOhy2CvUiy06p-Mew&s",
    },
    "axristo_dakilo": {
        "name": "axristo_dakilo",
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYcFlwRxQnvJGufTMBqIOhy2CvUiy06p-Mew&s",
    },
    "mesaio": {
        "name": "mesaio",
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYcFlwRxQnvJGufTMBqIOhy2CvUiy06p-Mew&s",
    },
    "mikro": {
        "name": "mikro",
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYcFlwRxQnvJGufTMBqIOhy2CvUiy06p-Mew&s",
    },
    "mpounia": {
        "name": "mpounia",
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYcFlwRxQnvJGufTMBqIOhy2CvUiy06p-Mew&s",
    },
}

# Global variables
buffer = []
latest_classification = {}
mqtt_connected = False
mqtt_data_received = False
last_message_time = 0
MESSAGE_TIMEOUT = 1

# Useful Paths
html_file_path = "./src/html/frontent.html"
model_path = "./MachineLearning/Binary_models/neural_network_(mlp)_model.pkl"
scaler_path = "./MachineLearning/Binary_models/scaler.pkl"

# MQTT configuration
mqtt_broker = "localhost"
mqtt_topic = "emg/sensor"
MQTT_USER = "Dimitris"
MQTT_PASSWORD = "Dimitris"
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

window_size = 1  # seconds
sample_rate = 1860  # Hz
buffer_size = int(window_size * sample_rate)

# Initialize scaler
scaler = StandardScaler()


# Load the model
def load_model(model_path: str, scaler_path: str):
    global model, scaler
    try:
        # Load the model
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        # Load the scaler
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")


# Call load_model once when starting the program
load_model(model_path=model_path, scaler_path=scaler_path)


# MQTT connection callbacks
def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    mqtt_connected = True
    print("MQTT Connected")


def on_disconnect(client, userdata, rc):
    global mqtt_connected, mqtt_data_received
    mqtt_connected = False
    mqtt_data_received = False
    print("MQTT Disconnected")


def on_message(client, userdata, message):
    global latest_classification, mqtt_data_received, last_message_time, buffer
    last_message_time = time.time()
    mqtt_data_received = True
    # Decode the incoming message and convert it to an integer
    data = int(message.payload.decode("utf-8"))
    buffer.append(data)
    # Check if the buffer has reached the required size
    if len(buffer) > buffer_size:
        # Perform classification when buffer is full
        prediction = classify_data(buffer)
        latest_classification = map_class_to_info(prediction)
        # Clear the buffer after classification
        buffer.clear()
        print(f"Updated classification: {latest_classification}")


# Configure MQTT client
mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message = on_message
mqtt_client.connect(mqtt_broker, 1883)
mqtt_client.subscribe(mqtt_topic)
mqtt_client.loop_start()


# Feature extraction from buffer
def extract_features(buffer):
    if len(buffer) < buffer_size:
        print("Buffer is too small for feature extraction.")
        return None
    # Extract features: RMS, MAV, and WL
    rms = np.sqrt(np.mean(np.square(buffer)))
    mav = np.mean(np.abs(buffer))
    wl = np.sum(np.abs(np.diff(buffer)))
    features = np.array([rms, mav, wl]).reshape(1, -1)
    if scaler.mean_ is None or scaler.scale_ is None:
        print("Scaler is not fitted.")
        return None
    # Scale the features
    scaled_features = scaler.transform(features)
    return scaled_features[0]


# Classify data based on features
def classify_data(buffer):
    features = extract_features(buffer)
    prediction = model.predict([features][0])
    return prediction


# Map prediction to class info
def map_class_to_info(prediction):
    return class_mapping.get(prediction)


# Read HTML content from file
def read_html_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Check if new data has been received
            if (
                mqtt_data_received
                and time.time() - last_message_time <= MESSAGE_TIMEOUT
            ):
                status = {"connected": True, "classification": latest_classification}
            else:
                status = {
                    "connected": False,
                    "classification": {"name": " ", "image": " "},
                }
            await websocket.send_json(status)
            await asyncio.sleep(0.25)  # Small delay
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


@app.get("/")
async def get():
    return HTMLResponse(content=read_html_content(html_file_path))
