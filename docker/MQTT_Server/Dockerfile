# Use the official Eclipse Mosquitto image as a base
FROM eclipse-mosquitto:latest

# Copy your custom configuration file and password file into the image
COPY ./config/mosquitto.conf /mosquitto/config/mosquitto.conf
COPY ./config/mosquitto.passwd /mosquitto/config/mosquitto.passwd

# Set the correct permissions for the password file (0700)
RUN chmod 0700 /mosquitto/config/mosquitto.passwd

# Set up volumes for persistence and logs
VOLUME ["/mosquitto/data", "/mosquitto/log"]

# Expose the MQTT default port
EXPOSE 1883

# Command to run the Mosquitto broker
CMD ["mosquitto", "-c", "/mosquitto/config/mosquitto.conf"]
