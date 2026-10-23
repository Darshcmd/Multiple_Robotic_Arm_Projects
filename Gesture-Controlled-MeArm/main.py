#!/usr/bin/env python3
"""Gesture-controlled MeArm-style arm. All five actuators move together from
live hand gestures (MediaPipe), or a synchronized verify sweep. Inspired by the
RoboticArm gesture script and gesture_MeArm (MediaPipe + serial).

Usage:
  ./venv/bin/python main.py                  # gesture control
  ./venv/bin/python main.py --sweep          # move all 5 together (verify)
  ./venv/bin/python main.py --no-hand         # static pose, no webcam
"""

import argparse, glob, json, signal, struct, sys, time
import cv2, numpy as np
import serial, serial.tools.list_ports
from hand_track import build_hands, close_hands, get_hand_state

PACKET_ORDER = ("arm1", "arm2", "tilt", "base", "claw")
SPACE_STEP = 45
CAM_DEFAULT = 0
EMA_ALPHA = 0.7  # smoothing factor for joint angles (0 = frozen, 1 = raw)
ACK_TIMEOUT = 0.5  # seconds to wait for the firmware ACK of a packet

# Set by request_stop() from the SIGTERM/SIGINT handler so the gesture loop can
# leave its while-loop and run its cleanup instead of being killed outright.
_STOP = False

# Device name fragments used by the common Arduino USB-serial chips.
ARDUINO_PORT_PATTERNS = (
    "/dev/cu.usbserial*",   # FTDI / Arduino Uno (official)
    "/dev/cu.usbmodem*",    # Arduino boards with native USB
    "/dev/cu.usb*",         # generic USB serial adapters
    "/dev/cu.SLAB*",        # CH340 / CP2102 clones
    "/dev/cu.wchusbserial*",
    "/dev/cu.SLAB_USBtoUART*",
    "/dev/tty.usbserial*",
)


def load_config(p):
    with open(p) as f:
        return json.load(f)


def find_arduino():
    """Find the Arduino serial port.

    Tries pyserial's enumeration first, then falls back to globbing the
    device tree, because macOS sometimes omits USB-serial adapters that are
    visible in /dev (notably CH340/CP2102 clones).
    """
    for p in serial.tools.list_ports.comports():
        if p.device.startswith(("/dev/cu.usb", "/dev/cu.SLAB", "/dev/tty.usb")):
            return p.device
    for pattern in ARDUINO_PORT_PATTERNS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def norm(v, lo, hi):
    return int(round(clamp(v, 0.0, 1.0) * (hi - lo) + lo))


def packet(angles):
    return struct.pack(">BBBBBB", 0xAA, angles["arm1"], angles["arm2"],
                       angles["tilt"], angles["base"], angles["claw"])


def connect(port, baud, timeout=2.5):
    """Open the serial port and read the firmware banner.

    Raises serial.SerialException if the port cannot be opened.
    """
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
    except Exception as e:
        raise serial.SerialException("cannot open {}: {}".format(port, e))
    time.sleep(2.5)
    banner = ""
    try:
        waiting = ser.in_waiting
        if waiting:
            banner = ser.read(waiting).decode(errors="replace").strip()
    except Exception:
        banner = ""
    return ser, banner


def map_hand(h, cfg):
    """Map normalized hand metrics (0..1) onto servo angles.

    Each joint uses its own [lo, hi] window from config.json. The claw window
    is [closed, open], so an open hand (claw metric -> 1) opens the gripper.
    """
    r = cfg["arm_limits"]
    return {j: norm(h[j], r[j][0], r[j][1]) for j in PACKET_ORDER}


def smooth(prev, cur, alpha=EMA_ALPHA):
    """EMA smoothing on joint angles to kill tracking jitter."""
    if prev is None:
        return dict(cur)
    return {j: int(round(alpha * cur[j] + (1.0 - alpha) * prev[j]))
            for j in PACKET_ORDER}


def send(ser, a):
    """Send one angle packet to the Arduino (fire-and-forget).

    Does NOT wait for an ACK, so the gesture loop can run at full camera
    frame rate without being stalled by serial round-trips. The ACK is still
    sent by the firmware and can be read separately if desired.
    """
    if ser is None:
        return ""
    try:
        ser.write(packet(a))
        return ""
    except serial.SerialException:
        return ""
    except (serial.SerialException, OSError) as e:
        print("Serial error while sending:", e)
        return ""


def waveform(i, total):
    half = total // 2
    if i <= half:
        return i / half
    return (total - 1 - i) / half


def sweep_all(ser, cfg, total=20):
    r = cfg["arm_limits"]
    print("\n=== All joints together (verify) ===")
    for i in range(total):
        if _STOP:
            print("  (interrupted - returning to mid pose)")
            break
        a = {}
        for j in PACKET_ORDER:
            a[j] = norm(waveform(i, total), r[j][0], r[j][1])
        ack = send(ser, a)
        print("  " + "  ".join("{} {}".format(j, a[j]) for j in PACKET_ORDER) + ("  ACK:" + ack if ack else ""))
        time.sleep(0.05)
    h = {}
    for j in PACKET_ORDER:
        h[j] = int(round((r[j][0] + r[j][1]) / 2.0))
    ack = send(ser, h)
    if ack:
        print("  HOLD:" + ack)
    print("Arm held at mid pose.")


def read_terminal_key():
    if not sys.stdin.isatty():
        return None
    try:
        import termios, tty, select
        old = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            return sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        return None
    except Exception:
        return None


def request_stop(signum=None, frame=None):
    """Ask the gesture loop to wind down and release the hardware.

    Installed as a SIGTERM/SIGINT handler so that killing the program (or a
    supervisor's timeout) still closes the camera, the tracker and the serial
    port instead of leaving the arm holding its last commanded pose.
    """
    global _STOP
    _STOP = True


def install_signal_handlers():
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass  # not on the main thread, or unsupported on this platform


def run_gesture(cfg, arduino, hands, cam=CAM_DEFAULT):
    # Accept both camera indices ("0", "1") and DroidCam URLs ("http://...")
    if isinstance(cam, str) and cam.isdigit():
        cam = int(cam)
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("Could not open webcam", cam, "- run with --no-hand")
        sys.exit(1)
    limits = cfg["arm_limits"]
    angles = {j: int(round((limits[j][0] + limits[j][1]) / 2.0)) for j in PACKET_ORDER}
    print("=== Gesture control (MediaPipe hand) ===")
    print("Camera", cam, "| keys: SPACE = base +45 | E = claw | Q = quit")
    last_sent = None
    smoothed = None
    base_offset = 0        # persistent offset added to the tracked base
    claw_override = None   # None = follow the hand, else a fixed angle
    quit_requested = False
    try:
        while not quit_requested and not _STOP:
            ok, frame = cap.read()
            if not ok:
                break
            h = get_hand_state(frame, hands)
            if h is not None:
                target = map_hand(h, cfg)
                smoothed = smooth(smoothed, target)
                angles = dict(smoothed)
            # SPACE adds a persistent offset so it survives hand tracking.
            if base_offset:
                angles["base"] = int(clamp(angles["base"] + base_offset,
                                           limits["base"][0], limits["base"][1]))
            if claw_override is not None:
                angles["claw"] = claw_override
            tk = read_terminal_key()
            if tk == " ":
                base_offset = (base_offset + SPACE_STEP) % 180
                angles["base"] = int(clamp(angles["base"] + SPACE_STEP,
                                           limits["base"][0], limits["base"][1]))
            elif tk in ("e", "E"):
                lo, hi = limits["claw"]
                if claw_override is None:
                    claw_override = hi
                elif claw_override == hi:
                    claw_override = lo
                else:
                    claw_override = None  # back to following the hand
            elif tk in ("q", "Q"):
                quit_requested = True
            if angles != last_sent:
                if arduino is not None:
                    send(arduino, angles)
                last_sent = dict(angles)
            cv2.putText(frame, "BASE={} ARM1={} ARM2={} TILT={} CLAW={}".format(
                angles["base"], angles["arm1"], angles["arm2"], angles["tilt"], angles["claw"]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "hand: {}{}".format(
                "yes" if h is not None else "no",
                "  base+" + str(base_offset) if base_offset else ""),
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if h is not None:
                cv2.putText(frame, "raw_claw={:.2f}".format(h["claw"]),
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                quit_requested = True
    finally:
        # Always release the camera, the tracker and the serial port, even if
        # the loop is interrupted by SIGTERM/SIGINT or an unexpected error.
        cap.release()
        cv2.destroyAllWindows()
        close_hands(hands)
        if arduino is not None:
            arduino.close()


def run_no_hand(cfg, arduino):
    angles = {j: int(round((cfg["arm_limits"][j][0] + cfg["arm_limits"][j][1]) / 2.0)) for j in PACKET_ORDER}
    print("=== No-hand mode: static pose (no webcam) ===")
    if arduino is not None:
        send(arduino, angles)
    cv2.namedWindow("Frame")
    try:
        while not _STOP:
            frame = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(frame, "HOME: " + "  ".join("{} {}".format(j, angles[j]) for j in PACKET_ORDER), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Gesture-controlled MeArm-style arm.")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--sweep", action="store_true", help="move all 5 joints together (verify)")
    ap.add_argument("--no-hand", action="store_true", help="static pose, no webcam")
    ap.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cam", type=str, default=str(CAM_DEFAULT), help="webcam index or DroidCam URL (e.g. http://192.168.1.50:4747/video)")
    ap.add_argument("--model", default=None, help="path to hand_landmarker.task")
    a = ap.parse_args()

    cfg = load_config(a.config)
    port = a.port or find_arduino()
    arduino = None
    install_signal_handlers()
    if port:
        try:
            arduino, banner = connect(port, a.baud, cfg["serial"].get("timeout", 2.5))
            print("Arduino on", port, "|", (banner or "no banner").splitlines()[0])
        except serial.SerialException as e:
            print("Cannot open", port, "-", e, "- running without serial movement")

    if a.sweep:
        if arduino is None:
            print("No Arduino found. Connect the board and re-run --sweep.")
            return 1
        try:
            sweep_all(arduino, cfg)
        finally:
            arduino.close()
        return 0

    if a.no_hand:
        try:
            return run_no_hand(cfg, arduino)
        finally:
            if arduino is not None:
                arduino.close()

    try:
        hands = build_hands(cfg, model_path=a.model)
    except Exception as e:
        print("MediaPipe not available:", e)
        print("Install it with: pip install -r requirements.txt")
        if arduino is not None:
            arduino.close()
        return 1

    try:
        run_gesture(cfg, arduino, hands, cam=a.cam)
    finally:
        # close_hands()/close() are idempotent, so the double close that
        # run_gesture performs is harmless.
        close_hands(hands)
        if arduino is not None:
            arduino.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

