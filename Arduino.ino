const int motionSensorPin = 2;  // HC-SR501 SENSOR
const int ledPin = 13;          // Built-in LED pin

void setup() {
    pinMode(motionSensorPin, INPUT);
    pinMode(ledPin, OUTPUT);
    Serial.begin(9600); // Initialize Serial Monitor
}

void loop() {
    int motionState = digitalRead(motionSensorPin);

    if (motionState == HIGH) {
        Serial.println("Motion Detected: YES\nLED is ON");
        digitalWrite(ledPin, HIGH);
    } else {
        Serial.println("Motion Detected: NO\nLED is OFF");
        digitalWrite(ledPin, LOW);
    }

    delay(500); 
}
