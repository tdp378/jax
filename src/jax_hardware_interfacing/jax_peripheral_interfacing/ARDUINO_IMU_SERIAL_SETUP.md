# Arduino IMU Serial Setup

This repo now expects the physical robot IMU to arrive over the existing Pi-to-Arduino serial link by default.

The Pi-side driver in [src/jax/scripts/jax_driver.py](../jax/scripts/jax_driver.py) listens for lines in this exact format:

```text
IMU:qx,qy,qz,qw,gx,gy,gz,ax,ay,az
```

Field order:
- `qx,qy,qz,qw`: quaternion orientation
- `gx,gy,gz`: angular velocity in radians/sec
- `ax,ay,az`: linear acceleration in meters/sec^2

## What Changed on the Pi Side

- Physical IMU defaults to serial in [src/jax/config/driver.yaml](../jax/config/driver.yaml).
- The main driver publishes `/imu/data` after parsing `IMU:` lines.
- The direct Pi-side BNO08x node is now opt-in only.

## Arduino Sketch Changes

If your current Arduino sketch is the servo controller sketch, add a BNO08x reader and print one `IMU:` line each loop iteration or on a fixed timer.

### Required Includes

Add these includes near the top of the sketch:

```cpp
#include <Adafruit_BNO08x.h>
#include <sh2/sh2.h>
#include <sh2/sh2_SensorValue.h>
```

### Global State

Add these globals near your other globals:

```cpp
Adafruit_BNO08x bno08x(-1);
sh2_SensorValue_t imuSensorValue;
bool imuReady = false;
unsigned long lastImuPublishMs = 0;
const unsigned long imuPublishIntervalMs = 20;

float imuQuatI = 0.0f;
float imuQuatJ = 0.0f;
float imuQuatK = 0.0f;
float imuQuatReal = 1.0f;
float imuGyroX = 0.0f;
float imuGyroY = 0.0f;
float imuGyroZ = 0.0f;
float imuAccelX = 0.0f;
float imuAccelY = 0.0f;
float imuAccelZ = 0.0f;
```

### IMU Init Helpers

Add these functions:

```cpp
void initImu() {
  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("IMU_INIT_FAIL");
    imuReady = false;
    return;
  }

  bno08x.enableReport(SH2_ROTATION_VECTOR);
  bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED);
  bno08x.enableReport(SH2_ACCELEROMETER);
  imuReady = true;
  Serial.println("IMU_READY");
}

void updateImu() {
  if (!imuReady) return;

  while (bno08x.getSensorEvent(&imuSensorValue)) {
    switch (imuSensorValue.sensorId) {
      case SH2_ROTATION_VECTOR:
        imuQuatI = imuSensorValue.un.rotationVector.i;
        imuQuatJ = imuSensorValue.un.rotationVector.j;
        imuQuatK = imuSensorValue.un.rotationVector.k;
        imuQuatReal = imuSensorValue.un.rotationVector.real;
        break;

      case SH2_GYROSCOPE_CALIBRATED:
        imuGyroX = imuSensorValue.un.gyroscope.x;
        imuGyroY = imuSensorValue.un.gyroscope.y;
        imuGyroZ = imuSensorValue.un.gyroscope.z;
        break;

      case SH2_ACCELEROMETER:
        imuAccelX = imuSensorValue.un.accelerometer.x;
        imuAccelY = imuSensorValue.un.accelerometer.y;
        imuAccelZ = imuSensorValue.un.accelerometer.z;
        break;
    }
  }
}

void publishImu() {
  if (!imuReady) return;
  if (millis() - lastImuPublishMs < imuPublishIntervalMs) return;
  lastImuPublishMs = millis();

  Serial.print("IMU:");
  Serial.print(imuQuatI, 6); Serial.print(',');
  Serial.print(imuQuatJ, 6); Serial.print(',');
  Serial.print(imuQuatK, 6); Serial.print(',');
  Serial.print(imuQuatReal, 6); Serial.print(',');
  Serial.print(imuGyroX, 6); Serial.print(',');
  Serial.print(imuGyroY, 6); Serial.print(',');
  Serial.print(imuGyroZ, 6); Serial.print(',');
  Serial.print(imuAccelX, 6); Serial.print(',');
  Serial.print(imuAccelY, 6); Serial.print(',');
  Serial.println(imuAccelZ, 6);
}
```

### Setup Changes

Call `initImu()` from `setup()` after `Wire.begin()`:

```cpp
  Wire.begin();
  initImu();
```

### Loop Changes

Call both IMU functions from `loop()`:

```cpp
  handleSerial();
  updateNeoPixels();
  updateBattery();
  updateImu();
  publishImu();
```

## Notes

- The sketch must keep the existing battery reply behavior for `BAT?` because the Pi still uses that.
- If your BNO08x address is not `0x4A`, change it in `begin_I2C(...)`.
- If the Arduino is too slow at 50 Hz, increase `imuPublishIntervalMs`.
- If IMU traffic ever interferes with command parsing, the next step is to move to a request/response IMU poll instead of free-running prints.
