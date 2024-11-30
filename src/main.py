import asyncio
import paho.mqtt.client as mqtt
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# FastAPI app initialization
app = FastAPI()

# Allow connections from all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin to access
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Generate synthetic data for movements (2 classes)
X, y = make_classification(
    n_samples=100,  # Number of samples
    n_features=1,  # One feature
    n_informative=1,  # One informative feature
    n_redundant=0,  # No redundant features
    n_classes=2,  # Two classes (adjusted for simplicity)
    n_clusters_per_class=1,  # One cluster per class
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train an SVM model
model = svm.SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)

# MQTT Broker settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USER = "Dimitris"
MQTT_PASSWORD = "Dimitris"

MOVEMENT_TOPIC = "gesture/movement"  # Topic for publishing movement
CONTROL_TOPIC = "gesture/control"  # Topic for controlling movement type

# Initialize MQTT client
client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.connect(MQTT_BROKER, MQTT_PORT, 120)

# Store received MQTT message data
received_data = []
current_movement = 0  # Default movement type

# Movement types for prediction and control
movement_types = {0: "Movement A", 1: "Movement B"}


# MQTT callback when message is received
def on_message(client, userdata, msg):
    global received_data
    payload = msg.payload.decode()
    received_data.append(payload)
    print(f"Received MQTT message: {payload}")


def on_control_message(client, userdata, msg):
    global current_movement
    payload = msg.payload.decode()
    try:
        current_movement = int(payload)
        print(
            f"Control movement updated to: {movement_types.get(current_movement, 'Unknown Movement')}"
        )
    except ValueError:
        print(f"Invalid control message: {payload}")


# Subscribe to MQTT topics
client.on_message = on_message
client.subscribe(MOVEMENT_TOPIC)
client.subscribe(CONTROL_TOPIC)  # Subscribe to control topic


# Run MQTT client loop in a separate thread
def mqtt_loop():
    client.loop_forever()


# Start the MQTT client loop in a separate thread
mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.daemon = True
mqtt_thread.start()


# Function to normalize EMG data
def normalize_data(data):
    return (data - 512) / 512  # Normalize for a range of 0 to 1023


# Function to simulate real-time SVM prediction and send movement updates
async def send_movement_updates(websocket: WebSocket):
    global current_movement

    previous_movement = None  # Keep track of the previous movement

    while True:
        # Wait for new data from MQTT
        if received_data:
            input_data = np.array([[normalize_data(float(received_data.pop(0)))]])

            prediction = model.predict(input_data)
            detected_movement = movement_types.get(current_movement, "Unknown Movement")

            # Publish only if movement changes
            if detected_movement != previous_movement:
                result = client.publish(
                    MOVEMENT_TOPIC, payload=detected_movement, qos=2, retain=True
                )
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    print(f"Published detected movement: {detected_movement}")
                previous_movement = detected_movement

            # Send the detected movement to the client (WebSocket)
            await websocket.send_text(detected_movement)

        # Sleep for 100 ms before sending next update
        await asyncio.sleep(0.1)


# WebSocket endpoint to handle real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # Accept the WebSocket connection
    try:
        # Start sending movement updates
        await send_movement_updates(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")


# To stop the MQTT client after completion (optional, for graceful shutdown)
def stop_mqtt():
    client.disconnect()
