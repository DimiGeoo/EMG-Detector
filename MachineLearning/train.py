"""
This file is responsible for collecting labeled data from MQTT messages and saving it to a CSV file
Input is the label and tracking for 3 minites repeatedly.
"""

import csv
import os
import time
from threading import Lock
from dotenv import load_dotenv
import paho.mqtt.client as mqtt

# Load environment variables from .env file
load_dotenv()

# MQTT connection details from environment variables
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = os.getenv("MQTT_TOPIC", "#")

# CSV_FILE = "MachineLearning/Datasets/labeled_data.csv"
CSV_FILE = "MachineLearning/Datasets/labeled_data_1.csv"
CSV_HEADERS = ["target", "feature"]

# Global variables
data_buffer = []  # Buffer for incoming data
lock = Lock()  # Lock for thread-safe operations


# Callback when connected to the MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"Failed to connect to MQTT broker. Return code: {rc}")


# Callback when a message is received
def on_message(_, __, msg):
    global data_buffer
    try:
        # Decode and convert the message to a float (assumes numeric payload)
        message = float(msg.payload.decode("utf-8"))
        with lock:
            data_buffer.append(message)
    except ValueError:
        print(f"Invalid message received: {msg.payload.decode('utf-8')}")


# Ensure the CSV file exists and has headers
def initialize_csv():
    try:
        with open(CSV_FILE, mode="x", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)
        print(f"Initialized CSV file: {CSV_FILE}")
    except FileExistsError:
        print(f"CSV file already exists: {CSV_FILE}")


# Collect data for a label and write to CSV after 3 Minites
def collect_and_save_label(label):
    global data_buffer
    labeled_data = []
    print(f"Collecting data for label '{label}' for 3 minites.")

    start_time = time.time()
    while time.time() - start_time < 60 * 3:
        with lock:
            labeled_data.extend(data_buffer)
            data_buffer.clear()
        time.sleep(0.5)  # Check the buffer every 500ms

    # Save labeled data to CSV
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([[label, value] for value in labeled_data])
        print(f"Saved {len(labeled_data)} data points for label '{label}' to CSV.")


# Main function
def main():
    initialize_csv()

    # Start the MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    # Configure MQTT credentials if provided
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    client.connect(MQTT_BROKER, MQTT_PORT, 60 * 3)  # 3 minutes timeout
    client.subscribe(TOPIC)
    client.loop_start()

    print("Listening for MQTT messages. You can label data dynamically.")

    try:
        while True:
            label = input(
                "Enter the label for the next 3 minites (or type 'exit' to stop): "
            ).strip()
            if label.lower() == "exit":
                break
            collect_and_save_label(label)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
