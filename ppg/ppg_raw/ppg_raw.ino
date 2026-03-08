const int PPG_PIN = A0;
const int SAMPLE_PERIOD = 10; // ms (100 Hz)

unsigned long lastSample = 0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  unsigned long now = millis();

  if (now - lastSample >= SAMPLE_PERIOD) {
    lastSample = now;

    int ppg = analogRead(PPG_PIN);

    Serial.print(now);
    Serial.print(",");
    Serial.println(ppg);
  }
}