#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_NeoPixel.h>
#include <utility/imumaths.h>
#include <math.h>

#define SERVO_COUNT 12
#define PWM_FREQ 50
#define SERVOMIN 150
#define SERVOMAX 600
#define PIXEL_PIN 6
#define PIXEL_COUNT 8
#define OE_PIN 7
#define BATTERY_PIN A6
#define IMU_I2C_ADDRESS 0x28

// --- MODES (Added NIGHT) ---
enum Mode { IDLE, STALKER, LOW_BAT, DISABLED, NIGHT };
Mode currentMode = IDLE;

const float RESISTOR_1 = 10000.0;
const float RESISTOR_2 = 2000.0;
const float DIVIDER_RATIO = (RESISTOR_1 + RESISTOR_2) / RESISTOR_2;
const float V_REF = 5.0;

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);
Adafruit_BNO055 bno055 = Adafruit_BNO055(55, IMU_I2C_ADDRESS, &Wire);
Adafruit_NeoPixel strip(PIXEL_COUNT, PIXEL_PIN, NEO_GRBW + NEO_KHZ800);

const uint8_t servoChannel[SERVO_COUNT] = {6, 7, 8, 9, 11, 12, 3, 4, 5, 0, 1, 2};
int directionSign[SERVO_COUNT] = {1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1};

String lineBuf = "";
float neo_t = 0.0;
unsigned long lastNeoUpdateMs = 0;
const unsigned long neoIntervalMs = 20;
float smoothedBatteryVolt = 0.0;
unsigned long lastBatUpdateMs = 0;
bool servosEngaged = false;
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

void(* resetFunc) (void) = 0; 

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

void writeServoDeg(int idx, int deg) {
  int pulse = map(deg, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(servoChannel[idx], 0, pulse);
}

void applyPoseRaw(int targetDeg[SERVO_COUNT]) {
  for (int i = 0; i < SERVO_COUNT; i++) {
    int d = 90 + directionSign[i] * (targetDeg[i] - 90);
    writeServoDeg(i, d);
  }
}

void enableServos(bool enable) {
  digitalWrite(OE_PIN, enable ? LOW : HIGH);
  servosEngaged = enable;
  if (!enable) currentMode = DISABLED;
  else if (currentMode == DISABLED) currentMode = IDLE;
}

void moveHome() {
  int home[SERVO_COUNT] = {90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90};
  applyPoseRaw(home);
}

void bootAnimation() {
  for(int i=0; i < PIXEL_COUNT; i++) {
    strip.setPixelColor(i, strip.Color(0, 150, 255, 0));
    strip.show();
    delay(60);
  }
  delay(100);
  for(int i=0; i<PIXEL_COUNT; i++) {
    strip.setPixelColor(i, strip.Color(150, 255, 255, 255));
  }
  strip.show();
  delay(100);
  for(int j=100; j>=0; j-=2) {
    float dec = j / 100.0;
    for(int p=0; p<PIXEL_COUNT; p++) {
      strip.setPixelColor(p, strip.Color(0, (uint8_t)(150 * dec), (uint8_t)(255 * dec), (uint8_t)(255 * dec)));
    }
    strip.show();
    delay(10);
  }
}

void updateNeoPixels() {
  if (millis() - lastNeoUpdateMs < neoIntervalMs) return;
  lastNeoUpdateMs = millis();
  
  float level = (sin(neo_t * 0.08) + 1.0) * 0.4 + 0.1;
  uint32_t c;

  switch(currentMode) {
    case STALKER:
      c = strip.Color((uint8_t)(255 * level), 0, 0, 0); 
      break;
    case LOW_BAT:
      if ((int)(neo_t) % 10 < 5) c = strip.Color(200, 150, 0, 0);
      else c = strip.Color(0, 0, 0, 0);
      break;
    case DISABLED:
      c = strip.Color(0, 0, 0, 20); 
      break;
    case NIGHT:
      // Solid bright white using the White channel (last param)
      c = strip.Color(0, 0, 0, 255); 
      break;
    case IDLE:
    default:
      c = strip.Color(0, (uint8_t)(150 * level), (uint8_t)(255 * level), 0); 
      break;
  }

  for (int i = 0; i < PIXEL_COUNT; i++) strip.setPixelColor(i, c);
  strip.show();
  neo_t += 1.0;
}

void updateBattery() {
  if (millis() - lastBatUpdateMs < 500) return;
  lastBatUpdateMs = millis();
  float instant = ((analogRead(BATTERY_PIN) * V_REF) / 1023.0) * DIVIDER_RATIO;
  if (smoothedBatteryVolt < 1.0) smoothedBatteryVolt = instant;
  else smoothedBatteryVolt = (smoothedBatteryVolt * 0.9) + (instant * 0.1);

  // Logic is commented out per your current preference
  // if (smoothedBatteryVolt < 6.6 && smoothedBatteryVolt > 1.0) currentMode = LOW_BAT;
}

bool parseAngles(String s, int out[SERVO_COUNT]) {
  s.trim();
  int lastPos = 0;
  for (int i = 0; i < SERVO_COUNT; i++) {
    int comma = s.indexOf(',', lastPos);
    String val;
    if (comma == -1 && i == SERVO_COUNT - 1) val = s.substring(lastPos);
    else if (comma != -1) { val = s.substring(lastPos, comma); lastPos = comma + 1; }
    else return false;
    out[i] = (int)round(val.toFloat());
  }
  return true;
}

void handleSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineBuf.length() > 0) {
        int angles[SERVO_COUNT];
        if (lineBuf == "HOME") {
          enableServos(true);
          moveHome();
          Serial.println("OK");
        } else if (lineBuf == "KILL") {
          enableServos(false);
          Serial.println("KILLED");
        } else if (lineBuf == "BAT?") {
          Serial.println(smoothedBatteryVolt);
        } else if (lineBuf == "REBOOT") {
          resetFunc();
        } else if (lineBuf == "MODE:STALKER") {
          currentMode = STALKER;
          Serial.println("MODE_RED");
        } else if (lineBuf == "MODE:IDLE") {
          currentMode = IDLE;
          Serial.println("MODE_CYAN");
        } else if (lineBuf == "MODE:NIGHT") {
          currentMode = NIGHT;
          Serial.println("MODE_WHITE");
        } else if (parseAngles(lineBuf, angles)) {
          if (!servosEngaged) enableServos(true);
          applyPoseRaw(angles);
          Serial.println("OK");
        } else {
          Serial.println("ERR");
        }
        lineBuf = "";
      }
    } else {
      lineBuf += c;
    }
  }
}

void setup() {
  pinMode(OE_PIN, OUTPUT);
  digitalWrite(OE_PIN, HIGH);
  Serial.begin(115200);
  Wire.begin();
  initImu();
  pwm.begin();
  pwm.setPWMFreq(PWM_FREQ);
  moveHome();
  strip.begin();
  strip.show();
  bootAnimation();
  Serial.println("JAX_READY");
}

void loop() {
  handleSerial();
  updateNeoPixels();
  updateBattery();
  updateImu();
  publishImu();
}