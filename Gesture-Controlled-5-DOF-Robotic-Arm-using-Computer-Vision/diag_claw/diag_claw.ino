// ============================================================================
// diag_claw.ino - Gripper (claw) calibration diagnostic, CHANNEL 5 ONLY.
//
// Purpose:
//   * Drive the 9g gripper servo with EXPLICIT microsecond pulses so the
//     PCA9685 oscillator/prescaler ambiguity cannot corrupt the timing.
//   * Discover the real safe OPEN and CLOSED pulse widths for the servo.
//   * Slew-limit every change so the mechanism is never slammed.
//
// Serial commands (9600 baud):
//   '+' / '-'  nudge command by 10 us      '[' / ']'  nudge by 1 us
//   'o'        jump to stored OPEN width    'c'        jump to stored CLOSED
//   'n'        jump to 1500 us (nominal centre)
//   'p'        print status                'i'        print servo limits
//   'w'        write OPEN/CLOSED to EEPROM 'r'        reload from EEPROM
//   'x'        stop pulsing (release)      'e'        resume pulsing
//
// SAFETY: the command is hard clamped to 500..2400 us, and the value on the
// wire is slewed toward the command by at most RAMP_STEP_US per RAMP_PERIOD_MS.
// ============================================================================
#include <Adafruit_PWMServoDriver.h>
#include <Wire.h>
#include <EEPROM.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// --- Gripper channel ------------------------------------------------------
static const uint8_t CH_CLAW = 5;

// --- HARD SAFETY LIMITS (microseconds) -------------------------------------
static const uint16_t US_HARD_MIN = 500;
static const uint16_t US_HARD_MAX = 2400;

// --- RAMP LIMITER: max change applied to the wire per step ----------------
static const uint16_t RAMP_STEP_US = 10;
static const uint16_t RAMP_PERIOD_MS = 40;  // => max 250 us/s slew rate

// --- EEPROM layout for the calibrated widths -------------------------------
static const int EE_OPEN = 0;    // uint16 (2 bytes)
static const int EE_CLOSED = 2;  // uint16 (2 bytes)

static uint16_t g_usCommand = 1500;  // requested by operator
static uint16_t g_usActual = 1500;   // currently on the wire
static uint16_t g_usOpen = 1200;     // calibrated "open" width
static uint16_t g_usClosed = 1800;   // calibrated "closed" width
static bool g_outputOn = true;
static unsigned long g_lastRamp = 0;

static uint16_t clampUs(int32_t v) {
  if (v < US_HARD_MIN) return US_HARD_MIN;
  if (v > US_HARD_MAX) return US_HARD_MAX;
  return (uint16_t)v;
}

static void applyPulse(uint16_t us) {
  pwm.writeMicroseconds(CH_CLAW, us);
}

static void loadCalib() {
  EEPROM.get(EE_OPEN, g_usOpen);
  EEPROM.get(EE_CLOSED, g_usClosed);
  g_usOpen = clampUs(g_usOpen);
  g_usClosed = clampUs(g_usClosed);
}

static void saveCalib() {
  EEPROM.put(EE_OPEN, g_usOpen);
  EEPROM.put(EE_CLOSED, g_usClosed);
}

static void printStatus() {
  uint16_t prescale = pwm.readPrescale();
  double usPerTick = 1000000.0 * (prescale + 1) / 25000000.0;
  double freq = 25000000.0 / ((prescale + 1.0) * 4096.0);
  uint16_t ticks = (uint16_t)(g_usActual / usPerTick + 0.5);

  Serial.print(F("cmd="));
  Serial.print(g_usCommand);
  Serial.print(F(" us  wire="));
  Serial.print(g_usActual);
  Serial.print(F(" us  ticks="));
  Serial.print(ticks);
  Serial.print(F("  prescale="));
  Serial.print(prescale);
  Serial.print(F("  us/tick="));
  Serial.print(usPerTick, 3);
  Serial.print(F("  pwm_freq="));
  Serial.print(freq, 2);
  Serial.print(F(" Hz  out="));
  Serial.print(g_outputOn ? F("ON") : F("OFF"));
  Serial.println();
}

static void printHelp() {
  Serial.println(F("+ - [ ] nudge | o open | c closed | n 1500 | p status"));
  Serial.println(F("i limits | w save EEPROM | r load EEPROM | x stop | e resume"));
}
void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Wire.begin();
  Serial.begin(9600);
  Serial.println(F("PCA9685 CLAW CALIB DIAG START"));

  // 25 MHz is the PCA9685 internal oscillator frequency (no crystal on this
  // board). Declaring 27 MHz here is what broke the original firmware.
  pwm.begin();
  pwm.setOscillatorFrequency(25000000);
  pwm.setPWMFreq(60);

  loadCalib();

  // Park at the stored open width, never at an arbitrary angle.
  g_usCommand = g_usOpen;
  g_usActual = g_usCommand;
  applyPulse(g_usActual);

  Serial.print(F("prescale="));
  Serial.println(pwm.readPrescale());
  printHelp();
  printStatus();
}
void loop() {
  // ---- ramp limiter -------------------------------------------------------
  if (g_outputOn && g_usActual != g_usCommand) {
    unsigned long now = millis();
    if (now - g_lastRamp >= RAMP_PERIOD_MS) {
      g_lastRamp = now;
      int32_t delta = (int32_t)g_usCommand - (int32_t)g_usActual;
      if (delta > RAMP_STEP_US) delta = RAMP_STEP_US;
      if (delta < -RAMP_STEP_US) delta = -RAMP_STEP_US;
      g_usActual = clampUs((int32_t)g_usActual + delta);
      applyPulse(g_usActual);
    }
  }

  // ---- serial commands ----------------------------------------------------
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    int32_t v;
    switch (c) {
      case '+': g_usCommand = clampUs((int32_t)g_usCommand + 10); break;
      case '-': g_usCommand = clampUs((int32_t)g_usCommand - 10); break;
      case ']': g_usCommand = clampUs((int32_t)g_usCommand + 1);  break;
      case '[': g_usCommand = clampUs((int32_t)g_usCommand - 1);  break;
      case 'n': g_usCommand = 1500; break;
      case 'o': g_usCommand = g_usOpen; break;
      case 'c': g_usCommand = g_usClosed; break;
      case 'p': printStatus(); break;
      case 'i':
        Serial.print(F("limits: hard "));
        Serial.print(US_HARD_MIN);
        Serial.print(F(".."));
        Serial.print(US_HARD_MAX);
        Serial.print(F(" us, open="));
        Serial.print(g_usOpen);
        Serial.print(F(" closed="));
        Serial.print(g_usClosed);
        Serial.println();
        break;
      case 'w':
        g_usOpen = clampUs(g_usOpen);
        g_usClosed = clampUs(g_usClosed);
        saveCalib();
        Serial.println(F("calibration saved to EEPROM"));
        break;
      case 'r':
        loadCalib();
        Serial.println(F("calibration reloaded from EEPROM"));
        printStatus();
        break;
      case 'x':
        g_outputOn = false;
        Serial.println(F("output STOPPED - servo released"));
        break;
      case 'e':
        g_outputOn = true;
        g_lastRamp = millis();
        Serial.println(F("output resumed"));
        break;
      case 'h': printHelp(); break;
      default: break;
    }
    v = 0; (void)v;
  }
}