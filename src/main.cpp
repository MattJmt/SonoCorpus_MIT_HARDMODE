#include <Arduino.h>

// Finger mapping (pinky excluded):
// A3 = Index, A2 = Middle, A1 = Ring, A0 = Thumb
constexpr uint8_t PIN_INDEX = A3;
constexpr uint8_t PIN_MIDDLE = A2;
constexpr uint8_t PIN_RING = A1;
constexpr uint8_t PIN_THUMB = A0;

// Pro Micro 5V model: ADC reference is AVCC (~5V) when using DEFAULT.
constexpr uint32_t SERIAL_BAUD = 115200;

void setup() {
  Serial.begin(SERIAL_BAUD);

  pinMode(PIN_INDEX, INPUT);
  pinMode(PIN_MIDDLE, INPUT);
  pinMode(PIN_RING, INPUT);
  pinMode(PIN_THUMB, INPUT);

  // Force 5V-range conversion on 5V boards (0..1023 maps to ~0..5V).
  analogReference(DEFAULT);

  // Throw away first conversion after reference setup for stability.
  (void)analogRead(PIN_INDEX);
  (void)analogRead(PIN_MIDDLE);
  (void)analogRead(PIN_RING);
  (void)analogRead(PIN_THUMB);

  delay(500);

  // Column labels are shown in Serial Plotter as channel names.
  Serial.println("Index\tMiddle\tRing\tThumb");
}

void loop() {
  const int indexValue = analogRead(PIN_INDEX);
  const int middleValue = analogRead(PIN_MIDDLE);
  const int ringValue = analogRead(PIN_RING);
  const int thumbValue = analogRead(PIN_THUMB);

  // 10-bit ADC: 1023 indicates input is at/above the selected reference.
  Serial.print(indexValue);
  Serial.print('\t');
  Serial.print(middleValue);
  Serial.print('\t');
  Serial.print(ringValue);
  Serial.print('\t');
  Serial.println(thumbValue);

  // ~200 Hz sample rate.
  delay(5);
}
