import os
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Lock

# Load environment variables from .env file
load_dotenv()

# MQTT connection details from environment variables
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = os.getenv("MQTT_TOPIC", "#")

# Data buffer for real-time plotting
data = []
lock = Lock()


# Callback when a message is received
def on_message(client, userdata, msg):
    global data
    try:
        value = float(msg.payload.decode("utf-8"))  # Assuming numeric payload
        with lock:
            data.append(value)
    except ValueError:
        print(f"Invalid data received: {msg.payload.decode('utf-8')}")


# Real-time plot updater
def update_plot(frame, line):
    with lock:
        line.set_ydata(data)
        line.set_xdata(range(len(data)))
    ax = line.axes
    ax.relim()
    ax.autoscale_view()  # Dynamically adjust axes to fit all data
    return (line,)


# Function to start MQTT client
def start_mqtt_client():
    client = mqtt.Client()
    client.on_message = on_message

    # Set username and password
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC)
    client.loop_start()
    return client


# Real-time plotting
fig, ax = plt.subplots()
ax.set_title("Real-Time EMG Signal")
ax.set_xlabel("Time (arbitrary units)")
ax.set_ylabel("Signal Value")
(line,) = ax.plot([], [], lw=2)

ani = FuncAnimation(
    fig, update_plot, fargs=(line,), interval=100
)  # Update every 0.1 sec

# Start MQTT client
mqtt_client = start_mqtt_client()

try:
    plt.show()  # Start the plotting loop
except KeyboardInterrupt:
    print("Exiting...")

# Stop MQTT client on exit
mqtt_client.loop_stop()
mqtt_client.disconnect()
