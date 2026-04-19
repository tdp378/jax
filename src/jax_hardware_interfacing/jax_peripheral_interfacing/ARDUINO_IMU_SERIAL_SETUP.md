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

If your current Arduino sketch is the servo controller sketch, add a BNO055 reader and print one `IMU:` line each loop iteration or on a fixed timer.

### Required Includes

Add these includes near the top of the sketch:

```cpp
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
```

### Global State

Add these globals near your other globals:

```cpp
Adafruit_BNO055 bno055 = Adafruit_BNO055(55, 0x28, &Wire);
bool imuReady = false;
unsigned long lastImuInitAttemptMs = 0;
const unsigned long imuInitRetryIntervalMs = 2000;
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
  unsigned long now = millis();
  if (imuReady || (now - lastImuInitAttemptMs) < imuInitRetryIntervalMs) {
    return;
  }

  lastImuInitAttemptMs = now;
  if (!bno055.begin()) {
    Serial.println("IMU_INIT_FAIL");
    imuReady = false;
    return;
  }

  bno055.setExtCrystalUse(true);
  imuReady = true;
  Serial.println("IMU_READY");
}

void updateImu() {
  if (!imuReady) {
    initImu();
    return;
  }

  imu::Quaternion quat = bno055.getQuat();
  imu::Vector<3> gyro = bno055.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  imu::Vector<3> linearAccel = bno055.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);

  imuQuatI = quat.x();
  imuQuatJ = quat.y();
  imuQuatK = quat.z();
  imuQuatReal = quat.w();

  imuGyroX = gyro.x();
  imuGyroY = gyro.y();
  imuGyroZ = gyro.z();

  imuAccelX = linearAccel.x();
  imuAccelY = linearAccel.y();
  imuAccelZ = linearAccel.z();
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
- If your BNO055 is strapped to `0x29` instead of `0x28`, change the constructor address.
- If the Arduino is too slow at 50 Hz, increase `imuPublishIntervalMs`.
- If IMU traffic ever interferes with command parsing, the next step is to move to a request/response IMU poll instead of free-running prints.
