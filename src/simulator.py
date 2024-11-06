import asyncio
import random
import paho.mqtt.client as mqtt
import threading

# MQTT Broker settings
MQTT_BROKER = "35def13f13c84ef0a86bf66859fa5b71.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "Dimitris"
MQTT_PASSWORD = "Dimitris"

# List of simulated gestures (topics)
GESTURE_TOPICS = [
    "gesture/thumb",
    "gesture/wave",
]


# Simulating EMG values (in range 0-1023)
def simulate_emg_reading():
    return random.randint(0, 1023)


# MQTT setup
client = mqtt.Client()

# Enable TLS (for secure MQTT communication)
client.tls_set()
client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Variable to keep track of pending messages
pending_messages = 0
lock = threading.Lock()

# Define the maximum number of unacknowledged messages (default is 20)
MAX_UNACKNOWLEDGED_MESSAGES = 20


# Callback for when a message has been published and acknowledged by the broker
def on_publish(client, userdata, mid):
    global pending_messages
    with lock:
        pending_messages -= 1  # Reduce the count of pending messages
    print(f"Acknowledged message ID: {mid}")


# Set the callback for message publish
client.on_publish = on_publish


# Run MQTT client loop in a separate thread
def mqtt_loop():
    client.loop_forever()


# Start the MQTT client loop in a separate thread
mqtt_thread = threading.Thread(target=mqtt_loop)
mqtt_thread.daemon = True
mqtt_thread.start()


# Function to asynchronously publish EMG sensor data for a specific gesture
async def publish_emg_data(gesture, message_count=20):
    global pending_messages

    for i in range(message_count):
        # Simulate an EMG reading
        emg_value = simulate_emg_reading()

        # Wait if there are too many unacknowledged messages
        while pending_messages >= MAX_UNACKNOWLEDGED_MESSAGES:
            await asyncio.sleep(0.1)  # Wait a little before retrying

        with lock:
            pending_messages += 1  # Increment the count of pending messages

        # Publish the simulated value to the corresponding gesture topic
        result = client.publish(gesture, payload=str(emg_value), qos=2, retain=True)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Published EMG value {emg_value} to topic {gesture}")
        else:
            print(f"Failed to publish EMG value {emg_value}, return code: {result.rc}")


# Asynchronous function to simulate EMG readings for all gestures
async def simulate_emg_sensor():
    tasks = []
    for gesture in GESTURE_TOPICS:
        tasks.append(
            publish_emg_data(gesture, message_count=50)
        )  # Adjust the message count as needed

    await asyncio.gather(*tasks)


# Start the asynchronous loop and publish EMG data
loop = asyncio.get_event_loop()
loop.run_until_complete(simulate_emg_sensor())

# Disconnect the MQTT client when done
client.disconnect()
