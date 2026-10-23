#!/usr/bin/env python3
"""Verify that main.py shuts down cleanly when it receives SIGTERM.

Launches the real gesture program, lets it run, then sends SIGTERM and
checks the exit status. Without a signal handler the process is killed
outright (exit code -15) and the camera/serial port are never released.
"""
import os
import signal
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "venv", "bin", "python")


def main_test():
    log = open("/tmp/sigterm_test.log", "wb")
    # stdin from /dev/null so the program does not try to read the terminal
    # (a background job would otherwise be suspended by job control).
    proc = subprocess.Popen(
        [PY, "-u", "main.py"],
        cwd=HERE, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print("started main.py, pid", proc.pid)
    time.sleep(12)
    print("sending SIGTERM ...")
    os.kill(proc.pid, signal.SIGTERM)
    try:
        rc = proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("FAIL  process did not exit within 10s of SIGTERM")
        proc.kill()
        return 1
    log.close()

    with open("/tmp/sigterm_test.log", "rb") as f:
        out = f.read().decode(errors="replace")
    acks = out.count("ACK: ACK")

    ok = True
    if rc == 0:
        print("PASS  exited cleanly on SIGTERM (rc=0)")
    elif rc == -signal.SIGTERM:
        print("FAIL  killed by SIGTERM without cleanup (rc={})".format(rc))
        ok = False
    else:
        print("PASS  exited cleanly on SIGTERM (rc={})".format(rc))
    if acks > 0:
        print("PASS  arm was driven before shutdown ({} ACKs)".format(acks))
    else:
        print("FAIL  no ACKs recorded - the arm never moved")
        ok = False
    if "Traceback" in out:
        print("FAIL  traceback in output")
        ok = False
    else:
        print("PASS  no traceback in output")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main_test())
