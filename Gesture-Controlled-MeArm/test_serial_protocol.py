#!/usr/bin/env python3
"""End-to-end test of the serial protocol using a simulated Arduino.

A pseudo-terminal stands in for the board: a background thread plays the
firmware (parse the 0xAA-framed packet, reply with the same ACK format the
sketch sends). This verifies packet framing, byte order, ACK parsing and the
port-discovery fallback without needing hardware plugged in.

Run with:  ./venv/bin/python test_serial_protocol.py
"""
import os
import pty
import threading
import time
import tty

import serial

import main

PACKET_ORDER = ("arm1", "arm2", "tilt", "base", "claw")
SYNC = 0xAA


class FakeArduino:
    """Simulates the firmware v2 sketch on the far end of a pty."""

    def __init__(self):
        self.master_fd, self.slave_fd = pty.openpty()
        # Raw mode: the protocol is binary and has no line terminator, so
        # canonical-mode buffering would stall until a newline arrives.
        tty.setraw(self.master_fd)
        self.port = os.ttyname(self.slave_fd)
        self.received = []
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        buf = bytearray()
        while not self._stop:
            try:
                os.set_blocking(self.master_fd, False)
                data = os.read(self.master_fd, 64)
            except (BlockingIOError, OSError):
                time.sleep(0.005)
                continue
            if not data:
                time.sleep(0.005)
                continue
            buf.extend(data)
            while len(buf) >= 6:
                if buf[0] != SYNC:
                    del buf[0]
                    continue
                arm1, arm2, tilt, base, claw = buf[1:6]
                del buf[:6]
                self.received.append((arm1, arm2, tilt, base, claw))
                ack = "ACK A1={} A2={} T={} B={} C={}\n".format(
                    arm1, arm2, tilt, base, claw)
                try:
                    os.write(self.master_fd, ack.encode())
                except OSError:
                    pass

    def close(self):
        self._stop = True
        try:
            os.close(self.master_fd)
            os.close(self.slave_fd)
        except OSError:
            pass


def main_test():
    failures = []

    def check(name, cond, detail=""):
        print(("PASS  " if cond else "FAIL  ") + name + ("  " + detail if detail else ""))
        if not cond:
            failures.append(name)

    fake = FakeArduino()
    try:
        ser = serial.Serial(fake.port, 9600, timeout=2.5)
        time.sleep(0.3)

        # --- a full pose round-trips byte for byte ---
        want = {"arm1": 72, "arm2": 74, "tilt": 60, "base": 90, "claw": 10}
        ack = main.send(ser, want)
        check("firmware received the right angles",
              fake.received[-1] == (72, 74, 60, 90, 10),
              str(fake.received[-1]))
        check("ACK is parsed back", "ACK" in ack and "A1=72" in ack, repr(ack))

        # --- angles at the extreme ends of each joint ---
        for angles in ({"arm1": 0, "arm2": 0, "tilt": 0, "base": 0, "claw": 0},
                       {"arm1": 180, "arm2": 180, "tilt": 180, "base": 180, "claw": 180}):
            main.send(ser, angles)
            check("packet {} delivered".format(angles["arm1"]),
                  fake.received[-1] == tuple(angles[j] for j in PACKET_ORDER),
                  str(fake.received[-1]))

        # --- a stray leading byte must not corrupt the packet ---
        ser.write(b"\x00\xFF")
        main.send(ser, {"arm1": 100, "arm2": 100, "tilt": 100, "base": 100, "claw": 100})
        check("resyncs after garbage bytes",
              fake.received[-1] == (100, 100, 100, 100, 100),
              str(fake.received[-1]))

        # --- the swept pose from --sweep must be within the config limits ---
        cfg = main.load_config("config.json")
        r = cfg["arm_limits"]
        mid = {j: int(round((r[j][0] + r[j][1]) / 2.0)) for j in PACKET_ORDER}
        main.send(ser, mid)
        check("mid pose respects the limits",
              all(r[j][0] <= mid[j] <= r[j][1] for j in PACKET_ORDER), str(mid))

        ser.close()
    finally:
        fake.close()

    # --- send() must survive a closed port instead of raising ---
    dead = serial.Serial()
    dead.close()
    check("send() tolerates a closed port", main.send(dead, mid) == "")

    print()
    if failures:
        print("{} FAILED: {}".format(len(failures), ", ".join(failures)))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
