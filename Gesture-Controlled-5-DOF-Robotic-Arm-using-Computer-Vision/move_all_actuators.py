#!/usr/bin/env python3
"""Move ALL robotic arm actuators at once (or one at a time).

Usage:
    ./venv/bin/python move_all_actuators.py                 # all together (default)
    ./venv/bin/python move_all_actuators.py --joint tilt    # all together except tilt

By default every servo moves simultaneously so you can watch all of them work
together. Pass --joint ARM1 (or any of the 5 names) to instead hold that one
joint still and sweep the other four together.

Auto-detects the Arduino (same rule as test_serial.py). Each packet prints the
frame and the ACK the firmware sends back.

Wire protocol (firmware v2):  0xAA + arm1, arm2, tilt, base, claw
PCA9685 channels:  BASE=0 ARM1=1 ARM2=2 TILT=4 CLAW=5   (CH_WRIST=3 is unused)
"""
import argparse
import struct
import sys

import serial
import serial.tools.list_ports
import time

PACKET_ORDER = ("arm1", "arm2", "tilt", "base", "claw")
LIMITS = {"arm1": (40, 140), "arm2": (40, 140), "tilt": (40, 140), "base": (30, 150), "claw": (110, 20)}
STARTUP = {"arm1": 72, "arm2": 74, "tilt": 60, "base": 90, "claw": 60}


def make_packet(arm1, arm2, tilt, base, claw):
    return struct.pack(">BBBBBB", 0xAA, arm1, arm2, tilt, base, claw)


def find_arduino():
    for p in serial.tools.list_ports.comports():
        if str(p.device).startswith(("/dev/cu.usb", "/dev/cu.usbserial", "/dev/cu.usbmodem")):
            return p.device
    return None


def send(ser, angles):
    ser.reset_input_buffer()
    ser.write(make_packet(**angles))
    time.sleep(0.15)
    ack = ""
    if ser.in_waiting:
        ack = ser.read(ser.in_waiting).decode(errors="replace").strip()
    return ack


def waveform(i, total):
    half = total // 2
    if i <= half:
        return i / half
    return (total - 1 - i) / half


def channel_number(j):
    return {"arm1": 1, "arm2": 2, "tilt": 4, "base": 0, "claw": 5}[j]


def sweep_all(ser, hold=None):
    order = [j for j in PACKET_ORDER if j != hold]
    end = {j: LIMITS[j][1] for j in order}
    total = 20
    print("\n=== All joints together" + ("" if not hold else " except " + hold.upper()) + " ===")
    for i in range(total):
        angles = dict(STARTUP)
        for j in order:
            angles[j] = round(STARTUP[j] + (end[j] - STARTUP[j]) * waveform(i, total))
        ack = send(ser, angles)
        line = " ".join("%s %s" % (k, v) for k, v in sorted(angles.items()))
        if ack:
            line += "  ACK: " + ack
        print("   " + line)
        time.sleep(0.05)
    print("Arm returned to safe startup pose: " + str(STARTUP))


def sweep_one(ser, joint):
    lo, hi = LIMITS[joint]
    if lo > hi:
        lo, hi = hi, lo
    print("\n=== Sweeping " + joint.upper() + " alone (CH " + str(channel_number(joint)) + ") ===")
    for pos in range(lo, hi + 1, 20):
        angles = dict(STARTUP)
        angles[joint] = pos
        ack = send(ser, angles)
        if ack:
            print("   ACK: " + ack)
        time.sleep(0.5)
    for pos in range(hi, lo - 1, -20):
        angles = dict(STARTUP)
        angles[joint] = pos
        ack = send(ser, angles)
        if ack:
            print("   ACK: " + ack)
        time.sleep(0.5)
    angles = dict(STARTUP)
    ack = send(ser, angles)
    if ack:
        print("   ACK: " + ack)
    print(joint + " back to center " + str(STARTUP[joint]))


def main():
    ap = argparse.ArgumentParser(description="Move all robotic arm actuators for verification.")
    ap.add_argument("--all", action="store_true", help="sweep every joint together (default)")
    ap.add_argument("joint", nargs="?", choices=PACKET_ORDER, help="hold this joint still and sweep all others")
    a = ap.parse_args()

    port = find_arduino()
    if port is None:
        print("NO ARDUINO FOUND - no /dev/cu.usb* device present.")
        print("Plug the Arduino in (v2 sketch flashed) then re-run.")
        sys.exit(1)

    print("Opening " + port + " ...")
    ser = serial.Serial(port, 9600)
    time.sleep(2.5)
    banner = ser.read(ser.in_waiting).decode(errors="replace").strip() if ser.in_waiting else ""
    print("Arduino connected on " + port + ". Firmware: " + (banner or "(no banner)"))
    send(ser, STARTUP)
    print("Arm set to safe startup pose: " + str(STARTUP))
    time.sleep(1.0)

    if a.joint:
        sweep_one(ser, a.joint)
    else:
        sweep_all(ser)

    ser.close()
    print("\nDONE. Watch every servo work together. Anything that doesn't move, moves the wrong way, or sounds strained means re-check its wiring.")


if __name__ == "__main__":
    main()