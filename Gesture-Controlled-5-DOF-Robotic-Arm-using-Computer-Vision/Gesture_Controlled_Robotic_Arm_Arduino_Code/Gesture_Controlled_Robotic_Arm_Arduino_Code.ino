// Gesture arm, PCA9685 version.
// Protocol v2: 0xAA sync plus angles.
// Sync stops bad byte alignment.
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

/*----- PCA9685 channel mapping (verified by diag sweep) -----*/
#define CH_BASE  0
#define CH_ARM1  1   // arm shoulder joint
#define CH_ARM2  2   // arm elbow joint
#define CH_WRIST 3
#define CH_TILT  4
#define CH_CLAW  5
/*----------------------------------*/

const byte SYNC_BYTE = 0xAA;

/*--- Per-channel angle limits (degrees), from config.json arm_limits ---*/
struct Limit { int lo; int hi; };
static const Limit CH_LIMITS[5] = {
  { 30, 150 },   // CH_BASE
  { 40, 140 },   // CH_ARM1
  { 40, 140 },   // CH_ARM2
  { 40, 140 },   // CH_TILT
  { 30, 90 },   // CH_CLAW (open/closed narrowed for safety)
};

/*--- Microsecond limits for the 9g servos (oscillator-independent) ---*/
static const uint16_t US_MIN = 500;   // 0 deg
static const uint16_t US_MAX = 2400;  // 180 deg

/* Safe startup positions (degrees) - never strain the gripper at boot */
static const int START_POS[5] = { 90, 90, 90, 90, 60 };

int base_angle  = START_POS[0];
int arm1_angle  = START_POS[1];
int arm2_angle  = START_POS[2];
int tilt_angle  = START_POS[3];
int claw_angle  = START_POS[4];

/* Map an angle (0..180) to microseconds (500..2400), then clamp to channel limits. */
int angleToUs(int channel, int angle) {
  int lo = CH_LIMITS[channel].lo;
  int hi = CH_LIMITS[channel].hi;
  angle = constrain(angle, lo, hi);
  int us = map(angle, 0, 180, US_MIN, US_MAX);
  return constrain(us, US_MIN, US_MAX);
}

void setChannelUs(uint8_t ch, int angle) {
  pwm.writeMicroseconds(ch, (uint16_t)angleToUs(ch, angle));
}

void setup() {
  Wire.begin();
  Serial.begin(115200);

  // 25 MHz is the PCA9685 internal oscillator (no crystal on this clone board).
  // The old 27 MHz value made all pulse widths wrong.
  pwm.begin();
  pwm.setOscillatorFrequency(25000000);
  pwm.setPWMFreq(60);
  delay(100);

  // Drive every channel to its safe startup position.
  setChannelUs(CH_BASE,  base_angle);
  setChannelUs(CH_ARM1,  arm1_angle);
  setChannelUs(CH_ARM2,  arm2_angle);
  setChannelUs(CH_TILT,  tilt_angle);
  setChannelUs(CH_CLAW,  claw_angle);

  Serial.println();
  Serial.println("ARM FIRMWARE V2 READY (5-angle: arm1,arm2,tilt,base,claw)");
}

void loop() {
  // find sync then read angles
  while (Serial.available()) {
    if (Serial.peek() != SYNC_BYTE) {
      Serial.read();          // discard until we find a sync byte
      continue;
    }
    if (Serial.available() < 6) break;  // wait for the full packet

    Serial.read();            // consume sync byte
    int arm1 = Serial.read();
    int arm2 = Serial.read();
    int tilt = Serial.read();
    int base = Serial.read();
    int claw = Serial.read();

    // sanity range check (belt and braces)
    if (arm1 <= 180 && arm2 <= 180 && tilt <= 180 && base <= 180 && claw <= 180) {
      arm1_angle = arm1;
      arm2_angle = arm2;
      tilt_angle = tilt;
      base_angle = base;
      claw_angle = claw;

      setChannelUs(CH_ARM1, arm1_angle);
      setChannelUs(CH_ARM2, arm2_angle);
      setChannelUs(CH_TILT, tilt_angle);
      setChannelUs(CH_BASE,  base_angle);
      setChannelUs(CH_CLAW,  claw_angle);

      // ACK confirms command delivery
      Serial.print("ACK A1=");  Serial.print(arm1_angle);
      Serial.print(" A2=");     Serial.print(arm2_angle);
      Serial.print(" T=");      Serial.print(tilt_angle);
      Serial.print(" B=");      Serial.print(base_angle);
      Serial.print(" C=");      Serial.println(claw_angle);
    }
  }
}
