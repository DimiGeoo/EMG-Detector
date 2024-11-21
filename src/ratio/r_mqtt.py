import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
DURATION = int(os.getenv("DURATION", 5))

MQTT_TOPIC = "#"

mqtt_sample_count = 0


# MQTT client callbacks
def on_connect(client, userdata, flags, rc):
    """
    Callback function for when the client connects to the broker.
    It subscribes to the specified MQTT topic.
    """
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)  # Subscribe to all topics


def on_message(client, userdata, msg):
    """
    Callback function for when a message is received.
    This counts the number of received messages.
    """
    global mqtt_sample_count
    mqtt_sample_count += 1


def on_disconnect(client, userdata, rc):
    """
    Callback function for when the client disconnects from the broker.
    """
    print(f"Disconnected with result code {rc}")


# Initialize MQTT client
client = mqtt.Client()

# Set username and password for MQTT
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

# Assign callbacks
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect


# Connect to the MQTT broker
def connect_mqtt():
    """
    Connects to the MQTT broker and starts the client loop.
    """
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)  # Connect to broker
        client.loop_start()  # Start the MQTT client loop in the background
        print(f"Connecting to {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        exit(1)


# Disconnect from the MQTT broker
def disconnect_mqtt():
    """
    Gracefully disconnects from the MQTT broker.
    """
    client.loop_stop()  # Stop the MQTT loop
    client.disconnect()  # Disconnect from the broker
    print("Disconnected from MQTT broker.")


# Function to get the number of messages received
def get_message_count():
    """
    Returns the number of messages received during the session.
    """
    global mqtt_sample_count
    return mqtt_sample_count


# Main function to run MQTT operations
if __name__ == "__main__":
    connect_mqtt()

    # Run for a specific duration to capture MQTT data (e.g., 60 seconds)
    try:
        start_time = time.time()

        while time.time() - start_time < DURATION:
            time.sleep(0.001)  # Wait for messages

        # Print the number of messages received
        print(f"\nMessage Received: {get_message_count()}")

        print(f"\nDuration: {DURATION}")

        print(f"Average : {(DURATION/get_message_count())*1000}ms\n")

    except KeyboardInterrupt:
        print("MQTT Client interrupted.")

    finally:
        disconnect_mqtt()
