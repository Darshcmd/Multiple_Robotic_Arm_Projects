# Gesture-Controlled MeArm (MediaPipe Hands)

A 5-DOF robotic arm driven by webcam hand tracking. The host PC runs MediaPipe's
Hand Landmarker, converts the hand pose into five normalized (0..1) metrics, maps
them onto servo angles, and streams them to an Arduino over a framed serial
protocol.

Joints: `arm1` (shoulder), `arm2` (elbow), `tilt` (wrist), `base` (rotation),
`claw` (gripper).

---

## Requirements

- Python 3.11 (3.9+ works, but 3.11 is what this was built against)
- An Arduino running the MeArm firmware (see "Firmware" below)
- A webcam

---

## Setup

### 1. Virtual environment

```bash
cd Gesture-Controlled-MeArm
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 2. Download the hand landmarker model

MediaPipe 1.x removed the old `mp.solutions.hands` API. This project uses the
Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`), which needs the
model file on disk:

```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

`main.py` looks for the model in `./models/hand_landmarker.task` relative to
this folder. Override it with `--model /path/to/hand_landmarker.task`.

### 3. Flash the Arduino

Upload the MeArm sketch to the board. It must speak the protocol below and send
an ACK per packet.

---

## Serial protocol

One packet is 6 bytes, sent as soon as the commanded angles change:

| Byte | Meaning |
|------|---------|
| 0    | `0xAA` sync byte |
| 1    | `arm1` angle (deg) |
| 2    | `arm2` angle (deg) |
| 3    | `tilt` angle (deg) |
| 4    | `base` angle (deg) |
| 5    | `claw` angle (deg) |

The firmware replies with an ACK, e.g. `ACK A1=72 A2=74 T=60 B=90 C=10`. The
host prints it, so you can confirm every movement actually reached the board.

---

## Usage

```bash
# Gesture control (webcam + MediaPipe)
./venv/bin/python main.py

# Move all five joints together in a synchronized sweep (verify wiring)
./venv/bin/python main.py --sweep

# Static pose, no webcam needed
./venv/bin/python main.py --no-hand
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.json` | Config file path |
| `--sweep` | off | Run the synchronized 5-joint verification sweep |
| `--no-hand` | off | Static pose mode, no webcam |
| `--port` | auto-detect | Serial port, e.g. `/dev/cu.usbserial-10` |
| `--baud` | `9600` | Serial baud rate |
| `--cam` | `0` | Webcam index |
| `--model` | `models/hand_landmarker.task` | Path to the hand landmarker model |

### Keys

| Key | Action |
|-----|--------|
| `SPACE` | Rotate the base by +45° (a persistent offset, so it survives hand tracking) |
| `E` | Cycle the claw: open → closed → follow the hand |
| `Q` | Quit |

---


## How the hand maps to the arm

`hand_track.py` computes five metrics, all in 0..1:

| Metric | Source | Meaning |
|--------|--------|---------|
| `base` | wrist `.x` | Hand left/right → base rotation |
| `arm1` | wrist `.y` | Hand up/down → shoulder |
| `arm2` | apparent palm size | Hand near/far → elbow |
| `tilt` | wrist → palm-centre direction | Forearm angle → wrist tilt |
| `claw` | pinch distance ÷ palm size | Fist → open hand → gripper |

The pinch and tilt metrics are **scale invariant** — they are normalized by the
palm size, so they stay stable when you move your hand toward or away from the
camera.

### Mapping to servo angles

`config.json` holds the per-joint travel window:

```json
"arm_limits": {
  "arm1": [40, 140],
  "arm2": [40, 140],
  "tilt":  [40, 140],
  "base":  [30, 150],
  "claw":  [10, 120]
}
```

A metric of `0.0` maps to the first value and `1.0` to the second. Note the
claw window is `[closed, open]` — so `10` is closed and `120` is open. Change
these to match your own servos.

---

## Calibration

The hand metrics depend on where the camera is and how you move, so calibrate
once for your setup:

- **`palm_min` / `palm_max`** — the apparent palm size (as a fraction of frame
  width) when your hand is as close / as far as you want `arm2` to reach.
- **`pinch_min` / `pinch_max`** — the pinch distance relative to palm size when
  your fingers are fully closed / fully spread.
- **`arm_limits`** — the real mechanical travel of each servo. Set these to the
  values that keep the arm clear of itself.

Start with wide windows, then tighten them so the arm uses its full travel
without hitting the stops.

---

## Port auto-detection

The Arduino is found by scanning `serial.tools.list_ports()` and then
globbing `/dev/cu.usb*`, `/dev/cu.SLAB*`, `/dev/tty.usb*`. The filesystem
fallback matters on macOS, where CH340/CP2102 clones are sometimes missing from
pyserial's enumeration. If the board is not found, the program still runs and
shows the tracking overlay, it just won't move the servos.

Override the port explicitly if detection picks the wrong device:

```bash
./venv/bin/python main.py --port /dev/cu.usbserial-10
```

---

## Tests

Three suites run without an Arduino or a camera:

```bash
./venv/bin/python test_metrics.py         # hand metrics, mapping, smoothing, framing
./venv/bin/python test_serial_protocol.py  # packet framing + ACK parsing vs a simulated board
./venv/bin/python test_gesture_loop.py     # gesture loop: keys, lost-hand hold, limits
```

`test_serial_protocol.py` stands up a pseudo-terminal that plays the firmware,
so it verifies the byte order and ACK parsing without hardware.

### Live checks (need the board / camera)

```bash
./venv/bin/python main.py --sweep          # all 5 joints through their full range
./venv/bin/python test_live_gesture.py 15  # stream your hand to the arm for 15s
./venv/bin/python test_sigterm_shutdown.py # SIGTERM -> clean exit, resources released
```

`--sweep` moves every joint together and prints the ACK for each pose.
`test_live_gesture.py` prints the tracked hand metrics, the mapped angles and the
firmware ACK for every packet, so you can confirm the arm follows your hand.
`test_sigterm_shutdown.py` starts the real program, terminates it with SIGTERM
and checks that it exits cleanly instead of being killed outright, which would
leak the camera and the serial port.

### Stopping the program

Press `Q` in the video window, or send `SIGTERM` / `Ctrl-C`. Both paths release
the camera, the hand tracker and the serial port, so the arm is left holding its
last commanded pose rather than being left mid-write.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point: CLI, serial link, gesture loop, sweep |
| `hand_track.py` | MediaPipe HandLandmarker wrapper and metric extraction |
| `config.json` | Joint limits and hand calibration |
| `models/hand_landmarker.task` | MediaPipe hand landmarker model (downloaded) |
| `test_metrics.py` | Metric and mapping tests |
| `test_serial_protocol.py` | Serial protocol tests |
| `test_gesture_loop.py` | Gesture control loop tests |
| `test_live_gesture.py` | Live hand → arm check (needs board + camera) |
| `test_sigterm_shutdown.py` | Clean shutdown on SIGTERM |
