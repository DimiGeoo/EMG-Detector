#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// Wi-Fi credentials
const char* ssid = "D*******";        
const char* password = "D*******"; 

// MQTT Broker details
const char* mqttServer = "192.168.1.5";  // IP Address of the MQTT Broker
const int mqttPort = 1883;                
const char* mqttUser = "D*******";      
const char* mqttPassword = "D*******"; 

const int detect_plus = D0;  
const int detect_minus = D1; 

// Topic
const char* mqttTopic = "emg/sensor";     

// EMG Sensor Pin
const int emgPin = A0; 

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  // Start serial communication
  Serial.begin(19200);
  pinMode(detect_plus, INPUT); 
  pinMode(detect_minus, INPUT); 

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

  if((digitalRead(detect_plus) == 1)||(digitalRead(detect_minus) == 1)){
    Serial.println("Not Connected");
    delay(1000);
  }
  else{
  // Send the value of Sensor to MQTT Broker
    client.publish(mqttTopic,  String(analogRead(emgPin)).c_str(),true,1); //QoS 1
  }
}

void reconnect() {
  // Reconnect to MQTT Broker
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "EMGClient-" + String(random(0xffff), HEX);

    if (client.connect(clientId.c_str(), mqttUser, mqttPassword)) {
      Serial.println("Connected to MQTT broker");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      delay(2000);
    }
  }
}
