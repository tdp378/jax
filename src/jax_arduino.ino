#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// =========================
// CONFIG
// =========================
#define SERVO_COUNT 12
#define PWM_FREQ 50
#define SERVOMIN 150
#define SERVOMAX 600
#define OE_PIN 7
#define BATTERY_PIN A6
#define SERVO_PACKET_TIMEOUT_MS 10
#define PI_SERIAL_BAUD 57600

// Preserve pre-I2C motion behavior: apply the same per-servo direction mapping
// that the previous firmware used.
#define USE_LEGACY_DIRECTION_MAPPING true

// Hardware Serial1 for Pi bridge (pins 0/RX, 1/TX).
#define piSerial Serial1

// Voltage divider
const float RESISTOR_1 = 10000.0;
const float RESISTOR_2 = 2000.0;
const float DIVIDER_RATIO = (RESISTOR_1 + RESISTOR_2) / RESISTOR_2;
const float V_REF = 5.0;

// =========================
// HARDWARE
// =========================
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40, Wire1);

// Servo mapping - physical channel order on PCA9685
const uint8_t servoChannel[SERVO_COUNT] = {6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 2, 1};
int directionSign[SERVO_COUNT] = {1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1};

// =========================
// STATE
// =========================
bool servosEngaged = false;
float batteryVolt = 0.0;
uint8_t rxCmd = 0;
uint8_t servoByteIndex = 0;
uint8_t servoPacket[SERVO_COUNT];
unsigned long servoPacketDeadlineMs = 0;
uint32_t piBytesWindow = 0;
uint32_t piCmd1Window = 0;
uint32_t piCmd2Window = 0;
uint32_t piCmd3Window = 0;
uint32_t piPoseAppliedWindow = 0;
uint32_t piPacketTimeoutWindow = 0;
unsigned long imuDebugLastMs = 0;

// =========================
// SERVO
// =========================
void writeServoDeg(int idx, int deg) {
  int pulse = map(deg, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(servoChannel[idx], 0, pulse);
}

void applyPose(int angles[SERVO_COUNT]) {
  // Keep legacy behavior unless explicitly disabled.
  for (int i = 0; i < SERVO_COUNT; i++) {
    int d = angles[i];
    if (USE_LEGACY_DIRECTION_MAPPING) {
      d = 90 + directionSign[i] * (angles[i] - 90);
    }
    writeServoDeg(i, d);
  }
}

void enableServos(bool enable) {
  digitalWrite(OE_PIN, enable ? LOW : HIGH);
  servosEngaged = enable;
}

void moveHome() {
  int home[SERVO_COUNT] = {90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90};
  applyPose(home);
}

// =========================
// BATTERY
// =========================
void updateBattery() {
  float raw = analogRead(BATTERY_PIN);
  batteryVolt = ((raw * V_REF) / 1023.0) * DIVIDER_RATIO;
}

// =========================
// SERIAL RECEIVE (commands from Pi)
// =========================
void processPiCommands() {
  while (piSerial.available() > 0) {
    piBytesWindow++;

    if (rxCmd == 0) {
      rxCmd = (uint8_t)piSerial.read();

      if (rxCmd == 1) {
        piCmd1Window++;
        servoByteIndex = 0;
        servoPacketDeadlineMs = millis() + SERVO_PACKET_TIMEOUT_MS;
      } else if (rxCmd == 2) {
        piCmd2Window++;
        moveHome();
        enableServos(true);
        rxCmd = 0;
      } else if (rxCmd == 3) {
        piCmd3Window++;
        enableServos(false);
        rxCmd = 0;
      } else {
        rxCmd = 0;
      }
      continue;
    }

    if (rxCmd == 1) {
      while (piSerial.available() > 0 && servoByteIndex < SERVO_COUNT) {
        servoPacket[servoByteIndex++] = (uint8_t)piSerial.read();
      }

      if (servoByteIndex >= SERVO_COUNT) {
        int newAngles[SERVO_COUNT];
        for (int i = 0; i < SERVO_COUNT; i++) {
          newAngles[i] = (int)servoPacket[i];
        }
        applyPose(newAngles);
        if (!servosEngaged) {
          enableServos(true);
        }
        piPoseAppliedWindow++;
        rxCmd = 0;
      } else if ((long)(millis() - servoPacketDeadlineMs) >= 0) {
        // Drop partial packets so stale bytes do not desync framing.
        piPacketTimeoutWindow++;
        rxCmd = 0;
        servoByteIndex = 0;
      }
    }
  }
}

// =========================
// SETUP
// =========================
void setup() {
  pinMode(OE_PIN, OUTPUT);
  digitalWrite(OE_PIN, HIGH);

  // USB debug.
  Serial.begin(115200);

  // Pi bridge on hardware Serial1 (D0/RX, D1/TX).
  piSerial.begin(PI_SERIAL_BAUD);

  // Servo driver on Qwiic I2C.
  Wire1.begin();
  pwm.begin();
  pwm.setPWMFreq(PWM_FREQ);

  Serial.println("JAX_SERIAL_READY");
}

// =========================
// LOOP
// =========================
void loop() {
  if (millis() - imuDebugLastMs >= 1000) {
    Serial.print("PI_DBG bytes_s=");
    Serial.print(piBytesWindow);
    Serial.print(" cmd1_s=");
    Serial.print(piCmd1Window);
    Serial.print(" cmd2_s=");
    Serial.print(piCmd2Window);
    Serial.print(" cmd3_s=");
    Serial.print(piCmd3Window);
    Serial.print(" pose_applied_s=");
    Serial.print(piPoseAppliedWindow);
    Serial.print(" pkt_timeout_s=");
    Serial.println(piPacketTimeoutWindow);
    piBytesWindow = 0;
    piCmd1Window = 0;
    piCmd2Window = 0;
    piCmd3Window = 0;
    piPoseAppliedWindow = 0;
    piPacketTimeoutWindow = 0;
    imuDebugLastMs = millis();
  }

  processPiCommands();

  static unsigned long lastTele = 0;
  if (millis() - lastTele > 500) {
    updateBattery();
    piSerial.print("B:");
    piSerial.println(batteryVolt);
    lastTele = millis();
  }
}
