#include "Arduino_RouterBridge.h"
#include <Wire.h>

#define MUX_ADDR 0x70
#define IMU_ADDR 0x6A

const uint8_t IMU_CHANNELS[] = {2, 4, 6, 7};

void selectMuxChannel(uint8_t ch) {
  Wire1.beginTransmission(MUX_ADDR);
  Wire1.write(1 << ch);
  Wire1.endTransmission();
}

void disableMux() {
  Wire1.beginTransmission(MUX_ADDR);
  Wire1.write(0x00);
  Wire1.endTransmission();
}

void initIMU(uint8_t ch) {
  selectMuxChannel(ch);
  delay(10);
  // Enable accel + gyro: ODR 104Hz, ±4g, ±250dps
  Wire1.beginTransmission(IMU_ADDR);
  Wire1.write(0x10); // CTRL1_XL
  Wire1.write(0x40); // 104Hz, ±4g
  Wire1.endTransmission();
  Wire1.beginTransmission(IMU_ADDR);
  Wire1.write(0x11); // CTRL2_G
  Wire1.write(0x40); // 104Hz, 250dps
  Wire1.endTransmission();
  disableMux();
}

String get_imu_data() {
  String result = "";
  for (uint8_t i = 0; i < 4; i++) {
    selectMuxChannel(IMU_CHANNELS[i]);
    delay(2);
    Wire1.beginTransmission(IMU_ADDR);
    Wire1.write(0x28); // OUTX_L_A
    Wire1.endTransmission(false);
    Wire1.requestFrom(IMU_ADDR, 6);
    int16_t ax = 0, ay = 0, az = 0;
    if (Wire1.available() == 6) {
      ax = Wire1.read() | (Wire1.read() << 8);
      ay = Wire1.read() | (Wire1.read() << 8);
      az = Wire1.read() | (Wire1.read() << 8);
    }
    disableMux();
    result += String(ax) + "," + String(ay) + "," + String(az);
    if (i < 3) result += ",";
    delay(2);
  }
  return result;
}

void setup() {
  Wire1.begin();
  delay(500);
  for (uint8_t i = 0; i < 4; i++) initIMU(IMU_CHANNELS[i]);
  Bridge.begin();
  Bridge.provide("get_imu_data", get_imu_data);
}

void loop() {}