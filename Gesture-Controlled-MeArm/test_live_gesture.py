#!/usr/bin/env python3
"""Live gesture check against the real Arduino and webcam.

Run this with your hand in front of the camera: it streams the tracked hand
metrics to the board and prints every commanded pose plus the firmware ACK,
so you can confirm the arm follows your hand.

Run with:  ./venv/bin/python test_live_gesture.py [seconds]
"""
import sys
import time

import cv2

import hand_track
import main


def main_test():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    cfg = main.load_config("config.json")

    port = main.find_arduino()
    if port is None:
        print("No Arduino found - connect the board first.")
        return 1
    ser, banner = main.connect(port, cfg["serial"].get("baud", 9600),
                               cfg["serial"].get("timeout", 2.5))
    print("board:", port, "|", (banner or "no banner").splitlines()[0])

    hands = hand_track.build_hands(cfg)
    cap = cv2.VideoCapture(main.CAM_DEFAULT)
    if not cap.isOpened():
        print("Could not open the webcam")
        return 1
    print("camera open. MOVE YOUR HAND INTO FRAME for {:.0f}s...".format(seconds))
    print("(no hand detected yet - showing what is being tracked)")

    smoothed = None
    last = None
    sends = 0
    seen = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        h = hands.process(frame)
        if h is None:
            continue
        seen += 1
        target = main.map_hand(h, cfg)
        smoothed = main.smooth(smoothed, target)
        if smoothed != last:
            ack = main.send(ser, smoothed)
            sends += 1
            last = dict(smoothed)
            print("  hand: " + str({k: round(v, 2) for k, v in h.items()}))
            print("    -> " + str({k: smoothed[k] for k in main.PACKET_ORDER}) +
                  "  ACK: " + (ack if ack else "(none)"))
        cv2.putText(frame, "hand: yes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    hand_track.close_hands(hands)
    ser.close()
    print()
    print("frames with a hand detected:", seen)
    print("packets sent to the arm:    ", sends)
    if seen == 0:
        print("No hand was detected - check the camera view and lighting.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
