void setup() {
  Serial1.begin(115200);
  delay(1000);
}

void loop() {
  Serial1.println("hello");
  delay(500);
}