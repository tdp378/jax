#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// =========================
// CONFIG
// =========================
#define I2C_ADDR 0x10
#define SERVO_COUNT 12
#define PWM_FREQ 50
#define SERVOMIN 150
#define SERVOMAX 600
#define OE_PIN 7
#define BATTERY_PIN A6

// Voltage divider
const float RESISTOR_1 = 10000.0;
const float RESISTOR_2 = 2000.0;
const float DIVIDER_RATIO = (RESISTOR_1 + RESISTOR_2) / RESISTOR_2;
const float V_REF = 5.0;

// =========================
// HARDWARE
// =========================
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40, Wire1);

// Servo mapping
const uint8_t servoChannel[SERVO_COUNT] = {6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 2, 1};
int directionSign[SERVO_COUNT] = {1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1};

// =========================
// STATE
// =========================
bool servosEngaged = false;
float batteryVolt = 0.0;

// =========================
// SERVO
// =========================
void writeServoDeg(int idx, int deg) {
  int pulse = map(deg, 0, 180, SERVOMIN, SERVOMAX);
  pwm.setPWM(servoChannel[idx], 0, pulse);
}

void applyPose(int angles[SERVO_COUNT]) {
  for (int i = 0; i < SERVO_COUNT; i++) {
    int d = 90 + directionSign[i] * (angles[i] - 90);
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
// I2C RECEIVE (commands from Pi)
// =========================
void onReceive(int len) {
  if (len < 1) return;

  uint8_t cmd = Wire.read();

  if (cmd == 1) {
    int angles[SERVO_COUNT];

    for (int i = 0; i < SERVO_COUNT; i++) {
      if (Wire.available()) {
        angles[i] = Wire.read();
      } else {
        angles[i] = 90;
      }
    }

    applyPose(angles);
    enableServos(true);
  } else if (cmd == 2) {
    moveHome();
    enableServos(true);
  } else if (cmd == 3) {
    enableServos(false);
  }
}

// =========================
// I2C REQUEST (Pi reads data)
// =========================
void onRequest() {
  updateBattery();

  // e.g. 742 = 7.42V
  uint16_t mv = (uint16_t)(batteryVolt * 100);
  Wire.write((uint8_t*)&mv, 2);
}

// =========================
// SETUP
// =========================
void setup() {
  pinMode(OE_PIN, OUTPUT);
  digitalWrite(OE_PIN, HIGH);

  Serial.begin(115200);

  // Pi I2C
  Wire.begin(I2C_ADDR);
  Wire.onReceive(onReceive);
  Wire.onRequest(onRequest);

  // Servo I2C
  Wire1.begin();
  pwm.begin();
  pwm.setPWMFreq(PWM_FREQ);

  Serial.println("JAX_I2C_READY");
}

// =========================
// LOOP
// =========================
void loop() {
  // nothing needed here
}
