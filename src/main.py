import logging
import ssl
import psycopg2
import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


app = FastAPI()

# Retrieve credentials from environment variables
mqtt_broker = os.getenv("MQTT_BROKER")
mqtt_port = os.getenv("MQTT_PORT")
mqtt_username = os.getenv("MQTT_USERNAME")
mqtt_password = os.getenv("MQTT_PASSWORD")
mqtt_topic = "#"
connection_uri = os.getenv("DATABASE_URL")  # in .env

# Check if required environment variables are set
required_env_vars = [
    mqtt_broker,
    mqtt_port,
    mqtt_username,
    mqtt_password,
    connection_uri,
]

if any(var is None for var in required_env_vars):
    raise ValueError("One or more required environment variables are missing.")

# Database connection setup
conn = psycopg2.connect(connection_uri)
cursor = conn.cursor()
cursor.execute("SET search_path TO public;")

# Create the mqtt_messages table if it doesn't exist
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS public.mqtt_messages (
        id SERIAL PRIMARY KEY,
        topic TEXT,
        message TEXT,
        qos INTEGER,
        retain BOOLEAN
    )
    """
)
conn.commit()


# MQTT callback for message handling
def on_message(client, userdata, msg):
    try:
        print(
            f"Received message from topic: {msg.topic}, message: {msg.payload.decode()}"
        )
        cursor.execute(
            """
            INSERT INTO mqtt_messages (topic, message, qos, retain)
            VALUES (%s, %s, %s, %s)
            """,
            (msg.topic, msg.payload.decode(), msg.qos, msg.retain),
        )
        conn.commit()
    except Exception as e:
        print(f"Error inserting message: {e}")
        conn.rollback()


# MQTT connection setup
def mqtt_connect():
    client = mqtt.Client()
    client.on_message = on_message
    client.username_pw_set(mqtt_username, mqtt_password)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker successfully")
            client.subscribe(mqtt_topic)
            print(f"Subscribed to topic: {mqtt_topic}")
        else:
            logging.error(f"Failed to connect, return code {rc}")

    client.on_connect = on_connect
    client.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2)

    try:
        client.connect(mqtt_broker, int(mqtt_port), 60)
        client.loop_start()
        print("Connecting to MQTT broker...")
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}")


mqtt_connect()


# FastAPI routes
@app.get("/")
def read_root():
    return {"message": "MQTT and FastAPI are working!"}


@app.get("/get_messages/")
async def get_messages():
    try:
        cursor.execute("SELECT * FROM mqtt_messages")
        rows = cursor.fetchall()
        return {"messages": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {e}")


# Cleanup on shutdown
@app.on_event("shutdown")
def shutdown():
    conn.close()
    print("Database connection closed")
