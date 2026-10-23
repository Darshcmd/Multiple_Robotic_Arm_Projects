#!/usr/bin/env python3
"""Sanity checks for the hand-tracking metric math and the angle mapping.

Run with:  ./venv/bin/python test_metrics.py
"""
import numpy as np

import hand_track
import main


class _P:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeResult:
    def __init__(self, lms):
        # The Tasks API exposes hand_landmarks as a list of landmark lists.
        self.hand_landmarks = lms


def synthetic_landmarks(wrist_x, wrist_y, pinch_frac, scale=80.0, w=640, h=640,
                        fore=(0.7071, 0.7071)):
    """Build 21 fake landmarks for a hand of palm length `scale` pixels.

    `scale` is the palm length in pixels, so the whole hand grows or shrinks
    together -- exactly what happens when a hand moves toward/away from the
    camera.

    `fore` is the unit forearm direction (wrist -> knuckles).
    """
    wx, wy = wrist_x * w, wrist_y * h
    fx, fy = fore
    px, py = -fy, fx  # perpendicular, for the knuckle spread

    lms = [_P(wrist_x, wrist_y)] * 21
    # Knuckles straddling the palm centre -> palm centre sits `scale` away.
    lms[5] = _P((wx + fx * scale + px * 0.5 * scale) / w,
                (wy + fy * scale + py * 0.5 * scale) / h)
    lms[9] = _P((wx + fx * scale - px * 0.5 * scale) / w,
                (wy + fy * scale - py * 0.5 * scale) / h)
    # Fingertips: same height, separated horizontally by pinch_frac * scale.
    tip_x = wx + fx * scale
    tip_y = wy + fy * scale - 0.8 * scale
    lms[8] = _P((tip_x - 0.5 * pinch_frac * scale) / w, tip_y / h)
    lms[12] = _P((tip_x + 0.5 * pinch_frac * scale) / w, tip_y / h)
    return lms


def main_test():
    cfg = main.load_config("config.json")
    ht = hand_track.build_hands(cfg)
    frame = np.zeros((480, 640, 3), np.uint8)
    failures = []

    def check(name, cond, detail=""):
        print(("PASS  " if cond else "FAIL  ") + name + ("  " + detail if detail else ""))
        if not cond:
            failures.append(name)

    try:
        # --- claw metric: a wide pinch must read higher than a tight pinch ---
        results = {}
        for pinch in (0.05, 0.30, 0.60):
            lms = synthetic_landmarks(0.5, 0.5, pinch, scale=80)
            ht._landmarker.detect_for_video = (
                lambda img, ts, _l=lms: _FakeResult([_l]))
            results[pinch] = ht.process(frame)["claw"]
        check("claw rises with pinch distance",
              results[0.05] < results[0.30] < results[0.60],
              str({k: round(v, 3) for k, v in results.items()}))
        check("tight pinch clamps to 0", results[0.05] == 0.0)

        # --- claw must be scale invariant (same pose, different hand depth) ---
        scaled = {}
        for scale in (40, 80, 160):
            lms = synthetic_landmarks(0.5, 0.5, 0.30, scale=scale)
            ht._landmarker.detect_for_video = (
                lambda img, ts, _l=lms: _FakeResult([_l]))
            scaled[scale] = ht.process(frame)["claw"]
        spread = max(scaled.values()) - min(scaled.values())
        check("claw is scale invariant", spread < 0.02,
              "spread={:.4f} {}".format(spread, {k: round(v, 3) for k, v in scaled.items()}))

        # --- arm2 must grow with apparent palm size ---
        sizes = {}
        for scale in (40, 80, 120):
            lms = synthetic_landmarks(0.5, 0.5, 0.30, scale=scale)
            ht._landmarker.detect_for_video = (
                lambda img, ts, _l=lms: _FakeResult([_l]))
            sizes[scale] = ht.process(frame)["arm2"]
        check("arm2 grows with palm size",
              sizes[40] < sizes[80] < sizes[120],
              str({k: round(v, 3) for k, v in sizes.items()}))

        # --- tilt must be scale invariant and track the forearm direction ---
        tilts = {}
        for scale in (40, 80, 160):
            lms = synthetic_landmarks(0.5, 0.4, 0.30, scale=scale)
            ht._landmarker.detect_for_video = (
                lambda img, ts, _l=lms: _FakeResult([_l]))
            tilts[scale] = ht.process(frame)["tilt"]
        spread = max(tilts.values()) - min(tilts.values())
        check("tilt is scale invariant", spread < 0.02,
              "spread={:.4f} {}".format(spread, {k: round(v, 3) for k, v in tilts.items()}))

        # Forearm pointing down -> high tilt, pointing up -> low tilt.
        for fore, expect_high in (((0.0, 1.0), True), ((0.0, -1.0), False)):
            lms = synthetic_landmarks(0.5, 0.5, 0.30, scale=80, fore=fore)
            ht._landmarker.detect_for_video = (
                lambda img, ts, _l=lms: _FakeResult([_l]))
            tilt = ht.process(frame)["tilt"]
            check("tilt direction (fore={})".format(fore),
                  (tilt > 0.7) if expect_high else (tilt < 0.3),
                  "tilt={:.3f}".format(tilt))

        # Horizontal forearm -> mid scale.
        lms = synthetic_landmarks(0.5, 0.5, 0.30, scale=80, fore=(1.0, 0.0))
        ht._landmarker.detect_for_video = (
            lambda img, ts, _l=lms: _FakeResult([_l]))
        tilt = ht.process(frame)["tilt"]
        check("tilt mid for horizontal forearm", 0.4 < tilt < 0.6,
              "tilt={:.3f}".format(tilt))

        # --- no hand detected -> None ---
        ht._landmarker.detect_for_video = lambda img, ts: _FakeResult([])
        check("no hand returns None", ht.process(frame) is None)

        # --- angle mapping uses the config windows ---
        full_open = main.map_hand({j: 1.0 for j in main.PACKET_ORDER}, cfg)
        full_closed = main.map_hand({j: 0.0 for j in main.PACKET_ORDER}, cfg)
        r = cfg["arm_limits"]
        check("map_hand honours limits (open)",
              all(full_open[j] == r[j][1] for j in main.PACKET_ORDER), str(full_open))
        check("map_hand honours limits (closed)",
              all(full_closed[j] == r[j][0] for j in main.PACKET_ORDER), str(full_closed))
        check("claw open angle is the larger value",
              r["claw"][1] > r["claw"][0], "claw window {}".format(r["claw"]))

        # --- packet framing ---
        pkt = main.packet({"arm1": 40, "arm2": 40, "tilt": 40, "base": 30, "claw": 10})
        check("packet is 6 bytes with sync 0xAA",
              len(pkt) == 6 and pkt[0] == 0xAA, pkt.hex())

        # --- smoothing moves toward the target but not straight to it ---
        lo = {j: r[j][0] for j in main.PACKET_ORDER}
        hi = {j: r[j][1] for j in main.PACKET_ORDER}
        s = main.smooth(lo, hi)
        check("smoothing is gradual",
              all(lo[j] < s[j] < hi[j] for j in main.PACKET_ORDER), str(s))
    finally:
        ht.close()

    print()
    if failures:
        print("{} FAILED: {}".format(len(failures), ", ".join(failures)))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
