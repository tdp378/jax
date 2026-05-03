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
#define I2C_RX_MAX 32

// Set true only if you explicitly need remote disable command support.
// Keeping this false prevents accidental limp state from malformed I2C traffic.
#define ALLOW_REMOTE_SERVO_DISABLE false

// Preserve pre-I2C motion behavior: apply the same per-servo direction mapping
// that the previous firmware used.
#define USE_LEGACY_DIRECTION_MAPPING true

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
volatile bool hasPendingPose = false;
volatile bool pendingEnableServos = false;
volatile bool pendingDisableServos = false;
int pendingAngles[SERVO_COUNT];

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
// I2C RECEIVE (commands from Pi)
// =========================
void onReceive(int len) {
  if (len <= 0) return;

  uint8_t rx[I2C_RX_MAX];
  int n = 0;
  while (Wire.available() && n < I2C_RX_MAX) {
    rx[n++] = (uint8_t)Wire.read();
  }

  if (n <= 0) return;

  const uint8_t cmd = rx[0];

  // Command 1 is binary_v1 servo command: [cmd=1, 12 angle bytes].
  if (cmd == 1) {
    int angleStart = -1;

    // Raw I2C payload shape: [1, a0..a11]
    if (n == (SERVO_COUNT + 1)) {
      angleStart = 1;
    }
    // SMBus block-write shape: [1, 12, a0..a11]
    else if (n == (SERVO_COUNT + 2) && rx[1] == SERVO_COUNT) {
      angleStart = 2;
    }

    if (angleStart < 0) {
      // Ignore malformed packets instead of applying partial/default angles.
      return;
    }

    for (int i = 0; i < SERVO_COUNT; i++) {
      pendingAngles[i] = (int)rx[angleStart + i];
    }

    hasPendingPose = true;
    pendingEnableServos = true;
    return;
  }

  // Other commands are single-byte controls.
  if (n != 1) {
    return;
  }

  if (cmd == 2) {
    for (int i = 0; i < SERVO_COUNT; i++) {
      pendingAngles[i] = 90;
    }
    hasPendingPose = true;
    pendingEnableServos = true;
  } else if (cmd == 3) {
    if (ALLOW_REMOTE_SERVO_DISABLE) {
      pendingDisableServos = true;
    }
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
  if (pendingDisableServos) {
    noInterrupts();
    pendingDisableServos = false;
    interrupts();
    enableServos(false);
  }

  if (hasPendingPose) {
    int localAngles[SERVO_COUNT];

    noInterrupts();
    for (int i = 0; i < SERVO_COUNT; i++) {
      localAngles[i] = pendingAngles[i];
    }
    hasPendingPose = false;
    bool shouldEnable = pendingEnableServos;
    pendingEnableServos = false;
    interrupts();

    applyPose(localAngles);
    if (shouldEnable) {
      enableServos(true);
    }
  }
}
