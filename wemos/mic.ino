#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// Wi-Fi credentials
const char* ssid = "*****";        // Replace with your Wi-Fi SSID
const char* password = "*****"; // Replace with your Wi-Fi Password

// MQTT Broker details
const char* mqttServer = "192.168.1.18";
const int mqttPort = 1883;                
const char* mqttUser = "Dimitris";      
const char* mqttPassword = "*****"; 

// Topics
const char* mqttTopic = "emg/sensor";     

// EMG Sensor Pin
const int emgPin = A0; 

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  // Start serial communication
  Serial.begin(19200);
  
  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi");

  // Connect to MQTT Broker
  client.setServer(mqttServer, mqttPort);
  while (!client.connected()) {
    Serial.print("Connecting to MQTT broker...");
    if (client.connect("EMGClient", mqttUser, mqttPassword)) {
      Serial.println("Connected to MQTT broker");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      delay(2000);
    }
  }
}

void loop() {
  // Ensure MQTT connection
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  int emgValue = analogRead(emgPin);

  // Publish the EMG value
  String emgValueStr = String(emgValue);
  client.publish(mqttTopic, emgValueStr.c_str());
  Serial.println(emgValue);

}

void reconnect() {
  // Reconnect to MQTT Broker
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("EMGClient", mqttUser, mqttPassword)) {
      Serial.println("Connected to MQTT broker");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      delay(2000);
    }
  }
}
