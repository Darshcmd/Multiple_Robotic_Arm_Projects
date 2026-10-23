#!/usr/bin/env python3
"""Headless test of the gesture-loop behaviour in main.run_gesture.

Stubs out the webcam, the hand tracker and OpenCV's window calls so the
control logic (smoothing, key handling, hold-on-lost-hand) can be verified
without a camera or an Arduino.

Run with:  ./venv/bin/python test_gesture_loop.py
"""
import cv2
import numpy as np

import main


class FakeTracker:
    """Hand tracker stub that returns a scripted sequence of hand states."""

    def __init__(self, script):
        self.script = list(script)
        self.index = 0

    def process(self, frame):
        if self.index >= len(self.script):
            return None
        state = self.script[self.index]
        self.index += 1
        return state

    def close(self):
        pass


class FakeCamera:
    """VideoCapture stub that always yields a blank frame."""

    def __init__(self, index):
        self.index = index
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.released = True


class DummyArduino:
    """Stand-in for the serial board so the loop actually issues packets."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def run_with(script, keys, cfg):
    """Run run_gesture with stubs and return the list of commanded poses."""
    poses = []
    tracker = FakeTracker(script)
    board = DummyArduino()

    orig_capture = cv2.VideoCapture
    orig_imshow = cv2.imshow
    orig_wait = cv2.waitKey
    orig_destroy = cv2.destroyAllWindows
    key_iter = iter(keys)
    frames = {"n": 0}

    cv2.VideoCapture = FakeCamera
    cv2.imshow = lambda *a, **k: None
    # The window key quits after the scripted number of frames is consumed.
    cv2.waitKey = lambda *a, **k: (ord("q") if frames["n"] >= len(script) else -1)
    cv2.destroyAllWindows = lambda *a, **k: None

    def fake_read_terminal_key():
        frames["n"] += 1
        return next(key_iter, None)

    orig_read_key = main.read_terminal_key
    main.read_terminal_key = fake_read_terminal_key

    orig_send = main.send
    main.send = lambda ser, a: poses.append(dict(a)) or ""

    try:
        main.run_gesture(cfg, board, tracker)
    finally:
        cv2.VideoCapture = orig_capture
        cv2.imshow = orig_imshow
        cv2.waitKey = orig_wait
        cv2.destroyAllWindows = orig_destroy
        main.read_terminal_key = orig_read_key
        main.send = orig_send
    return poses


def main_test():
    cfg = main.load_config("config.json")
    limits = cfg["arm_limits"]
    failures = []

    def check(name, cond, detail=""):
        print(("PASS  " if cond else "FAIL  ") + name + ("  " + detail if detail else ""))
        if not cond:
            failures.append(name)

    hand = {"base": 0.5, "arm1": 0.5, "arm2": 0.5, "tilt": 0.5, "claw": 1.0}

    # --- a tracked hand produces servo angles inside the limits ---
    poses = run_with([hand] * 6, [None] * 6, cfg)
    check("gesture loop produced poses", len(poses) > 0, "{} poses".format(len(poses)))
    check("all commanded angles respect the limits",
          all(limits[j][0] <= p[j] <= limits[j][1]
              for p in poses for j in main.PACKET_ORDER))
    check("open hand opens the claw", poses[-1]["claw"] == limits["claw"][1],
          "claw={}".format(poses[-1]["claw"]))

    # --- losing the hand holds the last pose instead of crashing ---
    poses = run_with([hand, hand, None, None], [None] * 4, cfg)
    check("holding pose when the hand is lost", len(poses) >= 1 and
          poses[-1]["arm1"] == poses[0]["arm1"],
          "{} -> {}".format(poses[0]["arm1"], poses[-1]["arm1"]))

    # --- SPACE applies a persistent base offset across frames ---
    # Compare against the same script without the key press: the difference
    # must be exactly one SPACE step for every pose that is emitted.
    script = [hand] * 4
    plain = run_with(script, [None] * 4, cfg)
    pressed = run_with(script, [" ", None, None, None], cfg)
    check("SPACE emits an offset base pose",
          pressed and pressed[0]["base"] != plain[0]["base"],
          "{} vs {}".format(pressed[0]["base"] if pressed else None,
                           plain[0]["base"]))
    check("SPACE offset survives hand tracking",
          all(p["base"] == b["base"] + main.SPACE_STEP
              for p, b in zip(pressed, plain)),
          "{} vs {}".format([p["base"] for p in pressed],
                           [b["base"] for b in plain]))
    check("base stays within limits",
          all(limits["base"][0] <= p["base"] <= limits["base"][1] for p in pressed))

    # --- E toggles the claw between open and closed, then back to the hand ---
    poses = run_with([hand] * 8, ["e", None, "e", None, "e", None, None, None], cfg)
    claws = [p["claw"] for p in poses]
    check("E opens the claw", limits["claw"][1] in claws, str(claws))
    check("E closes the claw", limits["claw"][0] in claws, str(claws))
    check("E returns to hand control", claws[-1] == limits["claw"][1], str(claws[-1]))

    # --- an interrupted run still releases the camera and the serial port ---
    # This mirrors the SIGTERM path: the loop must unwind through its finally
    # block instead of leaving the camera and the board open.
    released = {"camera": False, "board": False}
    tracker = FakeTracker([hand] * 100)

    orig_capture = cv2.VideoCapture
    orig_imshow = cv2.imshow
    orig_wait = cv2.waitKey
    orig_destroy = cv2.destroyAllWindows

    class StoppingCamera(FakeCamera):
        def release(self):
            released["camera"] = True
            super().release()

    class StoppingBoard(DummyArduino):
        def close(self):
            released["board"] = True
            super().close()

    board = StoppingBoard()

    cv2.VideoCapture = StoppingCamera
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1
    cv2.destroyAllWindows = lambda *a, **k: None
    orig_read_key = main.read_terminal_key
    main.read_terminal_key = lambda: None
    orig_send = main.send
    main.send = lambda ser, a: ""
    try:
        # request_stop() simulates the signal handler raising the stop flag.
        main.request_stop()
        main.run_gesture(cfg, board, tracker)
    finally:
        cv2.VideoCapture = orig_capture
        cv2.imshow = orig_imshow
        cv2.waitKey = orig_wait
        cv2.destroyAllWindows = orig_destroy
        main.read_terminal_key = orig_read_key
        main.send = orig_send
        main._STOP = False
    check("interrupted run releases the camera", released["camera"])
    check("interrupted run closes the serial port", released["board"])

    print()
    if failures:
        print("{} FAILED: {}".format(len(failures), ", ".join(failures)))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
