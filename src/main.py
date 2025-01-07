from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import random
from fastapi.staticfiles import StaticFiles
import paho.mqtt.client as mqtt
import time

# FastAPI App
app = FastAPI()

class_mapping = {
    0: {
        "name": "Class A",
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYcFlwRxQnvJGufTMBqIOhy2CvUiy06p-Mew&s",
    },
    1: {
        "name": "Class B",
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYcFlwRxQnvJGufTMBqIOhy2CvUiy06p-Mew&s",
    },
    2: {
        "name": "Class C",
        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYcFlwRxQnvJGufTMBqIOhy2CvUiy06p-Mew&s",
    },
}

latest_classification = {}
mqtt_connected = False
mqtt_data_received = False
last_message_time = 0
MESSAGE_TIMEOUT = 1

mqtt_broker = "localhost"
mqtt_topic = "#"
mqtt_client = mqtt.Client()
MQTT_USER = "Dimitris"
MQTT_PASSWORD = "Dimitris"
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)


def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    mqtt_connected = True
    print("MQTT Connected")


def on_disconnect(client, userdata, rc):
    global mqtt_connected, mqtt_data_received
    mqtt_connected = False
    print("MQTT Disconnected")


def on_message(client, userdata, message):
    global latest_classification, mqtt_data_received, last_message_time
    print(
        f"Received message on topic {message.topic}: {message.payload.decode('utf-8')}"
    )  # Debug log
    last_message_time = time.time()
    mqtt_data_received = True
    data = message.payload.decode("utf-8")
    prediction = classify_data(data)
    latest_classification = map_class_to_info(prediction)
    print(f"Updated classification: {latest_classification}")  # Debug log


mqtt_client.on_connect = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message = on_message
mqtt_client.connect(mqtt_broker, 1883)
mqtt_client.subscribe(mqtt_topic)
mqtt_client.loop_start()


def classify_data(data):
    prediction = random.randint(0, len(class_mapping) - 1)
    return prediction


def map_class_to_info(prediction):
    return class_mapping.get(prediction, {"name": "Unknown", "image": "unknown.png"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:  # Keep the WebSocket loop running
            # Check if new data has been received
            if (
                mqtt_data_received
                and time.time() - last_message_time <= MESSAGE_TIMEOUT
            ):
                status = {
                    "connected": True,
                    "classification": latest_classification,
                }
            else:
                status = {
                    "connected": False,
                    "classification": {"name": "Unknown", "image": "unknown.png"},
                }

            # Send status to WebSocket client
            await websocket.send_json(status)
            await asyncio.sleep(0.1)  # Small delay to prevent high CPU usage
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Classification</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
</head>
<body>
    <div class="container mt-5">
        <!-- Status Indicator -->
        <div id="status" class="alert alert-danger text-center" role="alert">
            Disconnected
        </div>

        <!-- Page Title -->
        <h1 class="text-center mb-4">Real-Time Classification</h1>

        <!-- Classification Display -->
        <div id="class-display" class="text-center p-4 border rounded shadow-sm">
            <p class="text-muted">Waiting for classification data...</p>
        </div>
    </div>

    <script>
        const socket = new WebSocket("ws://localhost:8000/ws");
        let previousConnectedState = false;

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateStatus(data.connected);
            if (data.connected) {
                updateUI(data.classification);
            }
        };

        function updateStatus(connected) {
            const statusDiv = document.getElementById("status");
            const classDisplay = document.getElementById("class-display");

            if (connected !== previousConnectedState) {
                previousConnectedState = connected;
                if (connected) {
                    statusDiv.textContent = "Connected";
                    statusDiv.className = "alert alert-success text-center"; // Bootstrap alert for success
                    classDisplay.style.display = "block"; // Show class display when connected
                } else {
                    statusDiv.textContent = "Disconnected";
                    statusDiv.className = "alert alert-danger text-center"; // Bootstrap alert for danger               
                    classDisplay.innerHTML = `<p class="text-muted">Waiting for classification data...</p>`;
                }
            }
        }

        function updateUI(classData) {
            const display = document.getElementById("class-display");
            if (classData && classData.name && classData.image) {
                display.innerHTML = `
                    <div class="d-flex flex-column align-items-center">
                        <img src="${classData.image}" alt="${classData.name}" class="img-fluid rounded mb-3">
                        <h2 class="fw-bold">${classData.name}</h2>
                    </div>
                `;
            } else {
                display.innerHTML = "<p class='text-muted'>Unknown classification</p>";
            }
        }
    </script>
</body>
</html>

"""


@app.get("/")
async def get():
    return HTMLResponse(content=html_content)
