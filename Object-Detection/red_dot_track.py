"""Red-dot detection + tracking -> smooth 5-DOF arm following.

The arm follows a red dot (laser / sticker) wherever it goes in the camera
frame. Learned from Gesture_Controlled_Robotic_Arm_Python_Code.py:
  - Gaussian blur before HSV, erode+dilate denoise
  - circularity-gated biggest-blob detector (rejects junk)
  - exponential (EMA) smoothing on pixels AND on joint angles
  - transmit only on change over framed serial (0xAA + 5 angle bytes)
  - vision-only fallback when no Arduino is plugged in

Safety / smoothness (never over-extend):
  - every angle clamped to red_config.json arm_limits (narrower than 0-180)
  - camera workspace cropped by margin so edge pixels can't slam end-stops
  - per-frame slew-rate limit (max degrees per frame) kills jerks
  - deadband: tiny pixel/angle changes are ignored
  - on target loss the arm HOLDS last pose (no snap-back, no hunting)

Usage:
    python red_dot_track.py [--config red_config.json] [--source 0] [--no-arm]
    Keys (terminal): E = claw | SPACE = pause/resume | Q = quit
    Mouse: click PAUSE / CLAW / QUIT buttons on the Frame window.
"""
import argparse
import json
import math
import os
import struct
import time

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "red_config.json")

# ---------- small helpers ----------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def lerp(a, b, t):
    return a + (b - a) * t


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def find_arduino():
    import serial.tools.list_ports
    for p in serial.tools.list_ports.comports():
        d = p.device
        if d.startswith("/dev/cu.usb") or d.startswith("/dev/cu.usbserial") \
                or d.startswith("/dev/cu.usbmodem") or d.startswith("/dev/ttyUSB") \
                or d.startswith("/dev/ttyACM") or d.startswith("COM"):
            return d
    return None


def open_arduino(port, baud=9600):
    import serial
    ar = serial.Serial(port, baud, timeout=0.5)
    time.sleep(2.5)  # Uno resets on open
    print("Arduino connected on", port)
    try:
        banner = ar.read(512).decode("ascii", "replace").strip()
    except Exception:
        banner = ""
    ar.timeout = 0  # non-blocking reads so the video loop never stalls
    if "ARM FIRMWARE V2" in banner:
        print("Firmware V2 protocol detected - matches the tracker.")
    elif banner:
        print("WARNING: unexpected firmware banner:", banner.splitlines()[0][:70])
        print("  Tracker speaks protocol V2 - if the arm ignores commands,")
        print("  re-flash Gesture_Controlled_Robotic_Arm_Arduino_Code.ino")
    else:
        print("NOTE: no firmware banner seen - if the arm never reacts,")
        print("  re-flash the V2 sketch (Gesture_Controlled_Robotic_Arm_Arduino_Code.ino).")
    return ar


def connect_arduino():
    """Open the first Arduino-looking port. Returns (serial_or_None, port_or_None)."""
    port = find_arduino()
    if port is None:
        return None, None
    try:
        return open_arduino(port), port
    except Exception as e:
        print(f"WARNING: cannot open {port} ({e})")
        return None, None


class SlewLimiter:
    """EMA + deadband + max-step limiter for one scalar channel."""

    def __init__(self, init, alpha, max_step, deadband):
        self.v = float(init)
        self.alpha = alpha
        self.max_step = max_step
        self.deadband = deadband

    def update(self, target):
        target = float(target)
        if abs(target - self.v) < self.deadband:
            return self.v
        ema = self.v + self.alpha * (target - self.v)
        dv = ema - self.v
        if abs(dv) > self.max_step:
            dv = math.copysign(self.max_step, dv)
        self.v += dv
        return self.v

# ---------- nested marker detector (red square card + dark dot) ----------
# Eye-in-hand: arm webcam looks at a salmon-red square card with a dark dot.
# Stage 1: find the card by dual-range red HSV + 4-sided polygon check.
# Stage 2: inside the card ROI, find the dark dot; its offset from the card
# center is the visual-servoing error. Falls back to card center if no dot.

def _squareness(c):
    """0..1 score: aspect x fill of the ROTATED min-area rect.
    Rotation-robust: the old approxPolyDP 4-vertex rule scored 0 for
    30-deg-rotated or blur-rounded cards (corners chamfer to 8 vertices).
    """
    area = cv2.contourArea(c)
    if area <= 0:
        return 0.0, None
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    if rw <= 0 or rh <= 0:
        return 0.0, None
    aspect = min(rw, rh) / max(rw, rh)   # 1 = square
    fill = area / (rw * rh)              # 1 = solid rect (any rotation)
    box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(int)
    return aspect * fill, box


def _best_square(mask, min_area, min_sq, frame_shape=None, border_margin=4):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    best, best_box, best_area = None, None, 0.0
    Hm, Wm = frame_shape if frame_shape else (None, None)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        # crop gate: a card cut by the frame edge touches the border. Direct
        # test, replaces the old axis-aligned fill gate which also rejected
        # legitimately rotated cards (30 deg rotation -> fill 0.54 < 0.70).
        if Hm is not None and (
                x <= border_margin or y <= border_margin
                or x + w >= Wm - border_margin or y + h >= Hm - border_margin):
            continue
        score, approx = _squareness(c)
        if score < min_sq:
            continue
        if area > best_area:
            best, best_box, best_area = c, approx, area
    return best, best_box, best_area


def _dot_in_roi(hsv_full, x1, y1, x2, y2, cfg):
    """Dark-dot blob inside card ROI. Returns (dx, dy) centroid or None."""
    H, W = hsv_full.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    roi = hsv_full[y1:y2, x1:x2]
    db = cfg.get("dot_hsv", {})
    lo = np.array(db.get("lower", [90, 100, 60]))
    hi = np.array(db.get("upper", [110, 255, 175]))
    dm = cv2.inRange(roi, lo, hi)
    dm = cv2.erode(dm, None, iterations=1)
    dm = cv2.dilate(dm, None, iterations=2)
    cnts = cv2.findContours(dm.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    min_area = cfg["tracking"].get("min_dot_area", 300)
    best, best_a = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * math.pi * a / (per * per) if per > 0 else 0
        if circ < 0.35:  # dot is roundish; card edges are not
            continue
        if a > best_a:
            best, best_a = c, a
    if best is None:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    return (x1 + int(M["m10"] / M["m00"]), y1 + int(M["m01"] / M["m00"]))


def detect_red_dot(frame_bgr, cfg):
    """Detect card + nested dot. Returns (cx, cy, size, mask, frame).

    (cx, cy) is the DOT centroid when visible (servo target), else the card
    center. Name kept so the pipeline below is untouched.
    """
    cam = cfg["camera"]
    hsv_cfg = cfg["red_hsv"]
    trk = cfg["tracking"]
    W = cam.get("frame_width", 800)
    frame = imutils.resize(frame_bgr, width=W)
    k = cam.get("blur_kernel", 7)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    l1 = np.array(hsv_cfg["lower1"]); u1 = np.array(hsv_cfg["upper1"])
    l2 = np.array(hsv_cfg["lower2"]); u2 = np.array(hsv_cfg["upper2"])
    mask = cv2.bitwise_or(cv2.inRange(hsv, l1, u1), cv2.inRange(hsv, l2, u2))
    mask = cv2.erode(mask, None, iterations=cam.get("erode_iter", 2))
    mask = cv2.dilate(mask, None, iterations=cam.get("dilate_iter", 2))

    best, best_box, best_area = _best_square(
        mask, trk.get("min_area", 400), trk.get("min_squareness", 0.45),
        frame_shape=mask.shape[:2])
    if best is None:
        return None, None, None, mask, frame
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None, None, None, mask, frame
    card_cx = int(M["m10"] / M["m00"])
    card_cy = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(best_box)
    size = float(max(w, h))
    if size < cam.get("min_radius_px", 6) * 2:
        return None, None, None, mask, frame

    # nested dot search, box shrunk 8% to skip fingers/edges
    mx, my = w // 12, h // 12
    dot = _dot_in_roi(hsv, x + mx, y + my, x + w - mx, y + h - my, cfg)
    cv2.drawContours(frame, [best_box], -1, (0, 255, 255), 2)
    if dot is not None:
        cv2.circle(frame, dot, 6, (255, 0, 0), -1)
        cv2.line(frame, (card_cx, card_cy), dot, (255, 0, 0), 2)
        return dot[0], dot[1], size, mask, frame
    cv2.circle(frame, (card_cx, card_cy), 5, (0, 165, 255), -1)
    return card_cx, card_cy, size, mask, frame


# ---------- eye-in-hand visual servoing ----------
# The camera rides ON the arm, so absolute pixel->angle mapping is wrong: the
# whole frame shifts whenever the arm moves. Instead we servo on the ERROR
# between the dot and the frame center: each loop nudges the joints a small
# step in the direction that shrinks the error (classic IBVS, one step/frame).
# - ex > 0 (dot right of center)  -> rotate base one way; ex < 0 -> other way
# - ey > 0 (dot below center)     -> pitch shoulder/elbow down; up -> up
# Gains (deg of joint per unit normalized error) + deadband live in config
# under "servo". All outputs still clamped to arm_limits.

def servo_step_targets(nx, ny, size_frac, cur, cfg):
    lim = cfg["arm_limits"]
    sv = cfg.get("servo", {})
    ex = nx - 0.5
    ey = ny - 0.5
    db = sv.get("deadband", 0.03)
    if abs(ex) < db:
        ex = 0.0
    if abs(ey) < db:
        ey = 0.0
    base = cur["base"] + sv.get("gain_base", 25.0) * ex
    arm1 = cur["arm1"] + sv.get("gain_arm1", 20.0) * ey
    arm2 = cur["arm2"] - sv.get("gain_arm2", 12.0) * ey
    # distance hold: grow/shrink reach to keep card apparent size constant
    if sv.get("size_hold", False):
        want = sv.get("target_size", 0.35)
        if size_frac is not None:
            ds = want - size_frac
            if abs(ds) > sv.get("size_deadband", 0.04):
                arm1 -= sv.get("gain_size", 15.0) * ds
                arm2 += sv.get("gain_size", 15.0) * ds * 0.6
    # wrist (tilt) stays out of the loop: base + arm1 + arm2 do the tracking.
    # gain_tilt (deg per unit ey, default 0) only nudges it if you opt in.
    tilt = cur["tilt"] + sv.get("gain_tilt", 0.0) * ey
    base = clamp(base, lim["base_min"], lim["base_max"])
    arm1 = clamp(arm1, lim["arm1_min"], lim["arm1_max"])
    arm2 = clamp(arm2, lim["arm2_min"], lim["arm2_max"])
    tilt = clamp(tilt, lim["tilt_min"], lim["tilt_max"])
    return base, arm1, arm2, tilt


def map_target_to_angles(nx, ny, cfg, size_frac=None, cur=None):
    """Back-compat wrapper.

    Eye-in-hand mode (cur given): servo step from current pose.
    Legacy static-cam mode (no cur): absolute mapping (kept for --mode static).
    """
    if cur is None:
        lim = cfg["arm_limits"]
        mp = cfg["mapping"]
        m = cfg["tracking"].get("workspace_margin", 0.05)
        nxc = clamp(nx, m, 1.0 - m)
        nyc = clamp(ny, m, 1.0 - m)
        nxc = (nxc - m) / (1.0 - 2 * m)
        nyc = (nyc - m) / (1.0 - 2 * m)
        b0, b1 = mp["base_for_nx_0"], mp["base_for_nx_1"]
        a1n0, a1n1 = mp["arm1_for_ny_0"], mp["arm1_for_ny_1"]
        a2n0, a2n1 = mp["arm2_for_ny_0"], mp["arm2_for_ny_1"]
        base = lerp(b0, b1, nxc)
        arm1 = lerp(a1n0, a1n1, nyc)
        arm2 = lerp(a2n0, a2n1, nyc) + 8.0 * (nxc - 0.5)
        # wrist fixed: only base/arm1/arm2 track the card (user request).
        tilt = mp.get("tilt_fixed", cfg["startup_pose"]["tilt"])
        base = clamp(base, lim["base_min"], lim["base_max"])
        arm1 = clamp(arm1, lim["arm1_min"], lim["arm1_max"])
        arm2 = clamp(arm2, lim["arm2_min"], lim["arm2_max"])
        tilt = clamp(tilt, lim["tilt_min"], lim["tilt_max"])
        return base, arm1, arm2, tilt
    return servo_step_targets(nx, ny, size_frac, cur, cfg)

# ---------- main loop ----------

def run(config_path, source, use_arm, show_mask, mode="static"):
    cfg = load_config(config_path)
    lim = cfg["arm_limits"]
    sm = cfg["smoothing"]
    trk = cfg["tracking"]
    pose = cfg["startup_pose"]
    eye = (mode == "eye-in-hand")

    arduino = None
    port = None
    if use_arm:
        arduino, port = connect_arduino()
        if arduino is None:
            import serial.tools.list_ports
            avail = [p.device for p in serial.tools.list_ports.comports()]
            print("WARNING: no Arduino found - VISION ONLY (arm will NOT move).")
            print("  USB serial ports visible:", ", ".join(avail) if avail else "(none)")
            print("  Plugging the board in later is fine - it auto-connects.")
    if arduino is not None:
        arm_status = f"ARM: LIVE ({port})"
        arm_col = (0, 255, 0)
        print("=" * 56)
        print(f"ARM CONNECTED on {port} - commands ARE being sent.")
        print("=" * 56)
    elif not use_arm:
        arm_status = "ARM: OFF (--no-arm)"
        arm_col = (0, 0, 255)
        print("ARM DISABLED by --no-arm - the arm will NOT move.")
    else:
        arm_status = "ARM: NOT FOUND"
        arm_col = (0, 0, 255)
        print("ARM NOT FOUND - plug in USB / check port; arm will NOT move.")

    px_alpha = sm.get("pixel_ema_alpha", 0.4)
    px_dead = sm.get("deadband_px", 4.0)
    smx = SlewLimiter(pose["base"], sm.get("angle_ema_alpha", 0.25),
                      sm.get("max_step_base", 3.0), sm.get("deadband_deg", 1.0))
    sm1 = SlewLimiter(pose["arm1"], sm.get("angle_ema_alpha", 0.25),
                      sm.get("max_step_arm1", 3.0), sm.get("deadband_deg", 1.0))
    sm2 = SlewLimiter(pose["arm2"], sm.get("angle_ema_alpha", 0.25),
                      sm.get("max_step_arm2", 3.0), sm.get("deadband_deg", 1.0))
    smt = SlewLimiter(pose["tilt"], sm.get("angle_ema_alpha", 0.25),
                      sm.get("max_step_tilt", 2.0), sm.get("deadband_deg", 1.0))

    sx, sy = None, None
    lost = 0
    paused = False
    claw = int(pose.get("claw", 10))
    last_tx = 0.0
    last_sent = None
    tx_gap = sm.get("tx_min_interval_s", 0.05)
    ack_count = 0
    writes_since_ack = 0
    no_ack_warned = False
    ack_announced = False
    last_port_scan = 0.0

    vs = VideoStream(src=source).start()
    time.sleep(2.0)
    cv2.namedWindow("Frame")
    BTN_PAUSE = (10, 80, 180, 128)
    BTN_CLAW = (190, 80, 360, 128)
    BTN_QUIT = (370, 80, 490, 128)
    quit_flag = {"q": False}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        nonlocal paused, claw
        if BTN_PAUSE[0] <= x <= BTN_PAUSE[2] and BTN_PAUSE[1] <= y <= BTN_PAUSE[3]:
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif BTN_CLAW[0] <= x <= BTN_CLAW[2] and BTN_CLAW[1] <= y <= BTN_CLAW[3]:
            claw = lim["claw_closed"] if claw != lim["claw_closed"] else lim["claw_open"]
            print(f"Claw -> {claw}")
        elif BTN_QUIT[0] <= x <= BTN_QUIT[2] and BTN_QUIT[1] <= y <= BTN_QUIT[3]:
            quit_flag["q"] = True

    cv2.setMouseCallback("Frame", on_mouse)
    print("SPACE pause | E claw | Q quit (terminal or buttons).")

    import sys as _sys
    try:
        import termios, tty, select
        _old = termios.tcgetattr(_sys.stdin)
        tty.setcbreak(_sys.stdin.fileno())
        stdin_ok = True
    except Exception:
        stdin_ok = False

    def term_key():
        if not stdin_ok:
            return None
        dr, _, _ = select.select([_sys.stdin], [], [], 0)
        return _sys.stdin.read(1) if dr else None

    try:
        while True:
            raw = vs.read()
            if raw is None:
                time.sleep(0.01)
                continue
            cx, cy, card_size, mask, frame = detect_red_dot(raw, cfg)
            H, W = frame.shape[:2]
            # frame-center crosshair: the servo goal in eye-in-hand mode
            cv2.drawMarker(frame, (W // 2, H // 2), (0, 255, 0),
                           cv2.MARKER_CROSS, 24, 2)

            if cx is not None and not paused:
                lost = 0
                if sx is None:
                    sx, sy = float(cx), float(cy)
                elif math.hypot(cx - sx, cy - sy) >= px_dead:
                    sx += px_alpha * (cx - sx)
                    sy += px_alpha * (cy - sy)
                nx, ny = sx / W, sy / H
                if eye:
                    cur = {"base": smx.v, "arm1": sm1.v,
                           "arm2": sm2.v, "tilt": smt.v}
                    size_frac = (card_size / W) if card_size else None
                    tB, t1, t2, tT = map_target_to_angles(
                        nx, ny, cfg, size_frac=size_frac, cur=cur)
                else:
                    tB, t1, t2, tT = map_target_to_angles(nx, ny, cfg)
                b = smx.update(tB); a1 = sm1.update(t1)
                a2 = sm2.update(t2); tt = smt.update(tT)
                ex, ey = nx - 0.5, ny - 0.5
                cv2.circle(frame, (int(sx), int(sy)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"TRACKING err {ex:+.2f},{ey:+.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                lost += 1
                b, a1, a2, tt = smx.v, sm1.v, sm2.v, smt.v
                msg = "SEARCHING..." if lost <= trk.get("max_lost_frames_before_freeze", 10) \
                    else "LOST - HOLDING POSE"
                col = (0, 165, 255) if lost <= 10 else (0, 0, 255)
                cv2.putText(frame, msg, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                if sx is not None:
                    cv2.circle(frame, (int(sx), int(sy)), 8, (0, 165, 255), 2)

            s1 = int(clamp(int(round(a1)), lim["arm1_min"], lim["arm1_max"]))
            s2 = int(clamp(int(round(a2)), lim["arm2_min"], lim["arm2_max"]))
            st = int(clamp(int(round(tt)), lim["tilt_min"], lim["tilt_max"]))
            sb = int(clamp(int(round(b)), lim["base_min"], lim["base_max"]))

            now = time.time()

            # --- hotplug: board missing at launch or unplugged mid-run ---
            if use_arm and arduino is None and (now - last_port_scan) >= 2.0:
                last_port_scan = now
                arduino, port = connect_arduino()
                if arduino is not None:
                    arm_status = f"ARM: LIVE ({port})"
                    arm_col = (0, 255, 0)
                    last_sent = None          # force a fresh pose packet
                    writes_since_ack = 0
                    print(f"Arduino (re)connected on {port}.")

            # --- drain firmware output: each "ACK ..." line = packet executed ---
            if arduino is not None:
                try:
                    n = arduino.in_waiting
                    if n:
                        acks = arduino.read(n).count(b"ACK")
                        if acks:
                            ack_count += acks
                            writes_since_ack = 0
                            if not ack_announced:
                                ack_announced = True
                                print("ACK received - the firmware is executing our packets.")
                except Exception as e:
                    print(f"Serial lost ({e}) - will auto-reconnect.")
                    try:
                        arduino.close()
                    except Exception:
                        pass
                    arduino = None
                    port = None
                    arm_status = "ARM: RECONNECTING - PLUG IN USB"
                    arm_col = (0, 165, 255)

            if (s1, s2, st, sb, claw) != last_sent and (now - last_tx) >= tx_gap:
                tag = "TX " if arduino is not None else "DRY-RUN"
                print(f"[{tag}] Arm1:{s1} Arm2:{s2} Tilt:{st} Base:{sb} Claw:{claw}")
                if arduino is not None:
                    try:
                        arduino.write(struct.pack(">BBBBBB", 0xAA, s1, s2, st, sb, claw))
                        writes_since_ack += 1
                        if writes_since_ack >= 5 and ack_count == 0 and not no_ack_warned:
                            no_ack_warned = True
                            print("WARNING: 5+ packets written but NO ACK came back.")
                            print("  Check: 1) V2 sketch flashed (banner 'ARM FIRMWARE V2 READY'),")
                            print("  2) no Serial Monitor / other program holding the port,")
                            print("  3) USB cable carries data (not charge-only).")
                    except Exception as e:
                        print("Serial write failed:", e)
                        arduino = None
                        port = None
                        arm_status = "ARM: RECONNECTING - PLUG IN USB"
                        arm_col = (0, 165, 255)
                last_sent = (s1, s2, st, sb, claw)
                last_tx = now

            for (x1, y1, x2, y2), label in (
                    (BTN_PAUSE, "PAUSE" if not paused else "RESUME"),
                    (BTN_CLAW, f"CLAW {claw}"),
                    (BTN_QUIT, "QUIT")):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1 + 8, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            info = f"nx {sx/W:.2f} ny {sy/H:.2f}" if sx is not None else "no dot"
            cv2.putText(frame, info, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            badge = arm_status
            badge_col = arm_col
            if arduino is not None:
                if ack_count:
                    badge += f" | ACK x{ack_count}"
                    badge_col = (0, 255, 0)
                elif writes_since_ack:
                    badge += f" | NO ACK ({writes_since_ack} pkts)"
                    badge_col = (0, 0, 255)
            elif use_arm:
                badge = "ARM: SEARCHING - PLUG IN USB"
                badge_col = (0, 165, 255)
            cv2.putText(frame, badge, (10, H - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, badge_col, 2)
            cv2.imshow("Frame", frame)
            if show_mask:
                mv = cv2.resize(mask, (400, int(mask.shape[0] * 400 / mask.shape[1])))
                cv2.imshow("RedMask (white = detected)", mv)

            key = cv2.waitKey(1) & 0xFF
            tk = term_key()
            if tk == " ":
                paused = not paused
            elif tk in ("e", "E"):
                claw = lim["claw_closed"] if claw != lim["claw_closed"] else lim["claw_open"]
            elif tk in ("q", "Q"):
                break
            if quit_flag["q"] or key == ord("q"):
                break
    finally:
        vs.stop()
        cv2.destroyAllWindows()
        if arduino is not None:
            try:
                arduino.close()
            except Exception:
                pass
        if stdin_ok:
            try:
                termios.tcsetattr(_sys.stdin, termios.TCSADRAIN, _old)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="Red-square + dot arm follower.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--source", type=int, default=0,
                    help="Camera index (laptop FaceTime is 0; C-port webcam was 1).")
    ap.add_argument("--no-arm", action="store_true")
    ap.add_argument("--show-mask", action="store_true")
    ap.add_argument("--mode", default="static", choices=["eye-in-hand", "static"],
                    help="static: absolute pixel->angle map (fixed laptop cam, default). "
                         "eye-in-hand: servo dot to frame center (arm cam: --source 1).")
    a = ap.parse_args()
    run(a.config, a.source, not a.no_arm, a.show_mask, mode=a.mode)


if __name__ == "__main__":
    main()

