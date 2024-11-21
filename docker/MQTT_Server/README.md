# EMG Detector MQTT Server 🚀

Details on using the MQTT server with Docker, utilizing the Mosquitto MQTT broker.

## ⚙️ Requirements

- **Docker**

## 📥 Setup & Installation

### Build the Docker Image

```bash
cd D:/GitHub/git/Personal/EMG-Detector/docker/MQTT_Server
docker build -t emg_mqtt .
```

## 🚀 Running the Server

### Start the MQTT Server

```bash
docker run -d --name emg_mqtt_container `
  -p 1883:1883 `
  -v ${PWD}\data:/mosquitto/data `
  -v ${PWD}\log:/mosquitto/log `
  emg_mqtt
```
```
docker run -d --name emg_mqtt_container -p 1883:1883 -v ${PWD}/data:/mosquitto/data -v ${PWD}/log:/mosquitto/log emg_mqtt
```

### Restart the Server

```bash
docker stop emg_mqtt_container
docker start emg_mqtt_container
```

## 🧪 Testing the Server

### Inside Docker

Access the container:

```bash
docker exec -it emg_mqtt_container /bin/sh
```

Subscribe to a topic:

```bash
mosquitto_sub -h localhost -t test/topic
```

Publish to a topic:

```bash
mosquitto_pub -h localhost -t test/topic -m "Hello, MQTT!"
```

### Outside Docker

Subscribe to a topic:

```bash
mosquitto_sub -h localhost -t test/topic
```

Publish to a topic:

```bash
mosquitto_pub -h localhost -t test/topic -m "Hello, MQTT!"
```

