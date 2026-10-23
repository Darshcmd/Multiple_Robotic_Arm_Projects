"""Train the marker HSV filters from hex colours (+ optional live refinement).

Marker = salmon-red square card with a dark dot in the center (eye-in-hand:
the webcam rides on the arm and the tracker servos the dot to frame center).

Usage:
    python train_red.py --hex "#E8908A"              # card colour only
    python train_red.py --hex "#E8908A" --dot-hex "#2F4A55"   # card + dot
    python train_red.py --hex FF0000 --refine --frames 60
    python train_red.py --hex "#FF0000" --image ./dot.jpg
    python train_red.py --hex "#FF0000" --show

1. Converts hex (sRGB) to OpenCV HSV.
2. Builds a DUAL HSV range for the card (red wraps at hue 0).
3. Optionally builds a SINGLE range for the dark dot (--dot-hex).
4. Optionally refines S/V floors from live camera (--refine) or image.
5. Writes red_hsv (+ dot_hsv) into red_config.json for red_dot_track.py.

Mirrors auto_hsv.py / hsv_tune.py idea from the gesture repo, but for the
nested card+dot marker driven by user-supplied hex colours.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "red_config.json")


def hex_to_bgr(hex_str):
    h = hex_str.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f'Bad hex "{hex_str}" - expected RRGGBB like #FF1A1A.')
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        raise ValueError(f'Bad hex "{hex_str}" - not valid hex digits.')
    return (b, g, r)


def bgr_to_hsv_pixel(bgr):
    px = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def bounds_from_hsv(h, s, v, hue_tol=8, sat_min=110, val_min=70, single=False):
    """HSV bounds for a colour. Dual range (red wrap) unless single=True.

    Returns (l1,u1,l2,u2) for dual, or (lo,hi) for single (dot colours
    don't wrap, so one range is enough and rejects more background).
    """
    s_lo = max(sat_min, min(s, 255) - 60)
    v_lo = max(val_min, min(v, 255) - 60)
    if single:
        lo = [max(0, h - hue_tol), s_lo, v_lo]
        hi = [min(179, h + hue_tol), 255, 255]
        return lo, hi
    if h < 20:  # normal reds near hue 0
        hi1 = min(179, max(10, h + hue_tol))
        lower1, upper1 = [0, s_lo, v_lo], [hi1, 255, 255]
        lower2, upper2 = [165, s_lo, v_lo], [180, 255, 255]
    else:  # deep/crimson reds living in high wrap zone
        lo2 = max(150, h - hue_tol)
        lower1, upper1 = [0, s_lo, v_lo], [10, 255, 255]
        lower2, upper2 = [lo2, s_lo, v_lo], [180, 255, 255]
    return lower1, upper1, lower2, upper2


def refine_from_frames(frames_bgr, target_h, hue_tol=8):
    """Tighten S/V floors from real pixels near the target hue."""
    hsv_stack = []
    for f in frames_bgr:
        blur = cv2.GaussianBlur(f, (7, 7), 0)
        hsv_stack.append(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).reshape(-1, 3))
    pix = np.concatenate(hsv_stack, axis=0)
    h_all = pix[:, 0].astype(int)
    s_all = pix[:, 1].astype(int)
    v_all = pix[:, 2].astype(int)
    dh = np.minimum(np.abs(h_all - target_h), 180 - np.abs(h_all - target_h))
    near = (dh <= hue_tol + 4) & (s_all >= 60) & (v_all >= 50)
    count = int(near.sum())
    if count < 800:
        return None, count
    s_lo = max(40, int(np.percentile(s_all[near], 3)) - 20)
    v_lo = max(30, int(np.percentile(v_all[near], 3)) - 20)
    return (s_lo, v_lo, count), count
def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def save_config(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def grab_camera_frames(source=0, n=60, width=800):
    from imutils.video import VideoStream
    import time
    import imutils
    vs = VideoStream(src=source).start()
    time.sleep(2.0)
    frames = []
    try:
        for _ in range(n):
            fr = vs.read()
            if fr is None:
                time.sleep(0.05)
                continue
            frames.append(imutils.resize(fr, width=width))
            time.sleep(0.05)
    finally:
        vs.stop()
    return frames


def main():
    ap = argparse.ArgumentParser(description="Train marker HSV bounds from hex.")
    ap.add_argument("--hex", required=True, help='Card colour, e.g. "#E8908A" or FF0000')
    ap.add_argument("--dot-hex", default=None, help='Dot colour, e.g. "#2F4A55" (single range, no wrap)')
    ap.add_argument("--dot-hue-tol", type=int, default=10)
    ap.add_argument("--hue-tol", type=int, default=8)
    ap.add_argument("--sat-min", type=int, default=30,
                    help="Lower for salmon cards (default 30; old bright-red default was 110).")
    ap.add_argument("--val-min", type=int, default=70)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--source", type=int, default=1,
                    help="Camera index (arm webcam is 1).")
    ap.add_argument("--image", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--config", default=CONFIG_PATH)
    args = ap.parse_args()

    bgr = hex_to_bgr(args.hex)
    h, s, v = bgr_to_hsv_pixel(bgr)
    print(f"Card hex {args.hex} -> BGR{bgr} -> HSV({h},{s},{v})")
    l1, u1, l2, u2 = bounds_from_hsv(h, s, v, args.hue_tol, args.sat_min, args.val_min)
    print(f"Card bounds: l1 {l1} u1 {u1} / l2 {l2} u2 {u2}")

    dot_bounds = None
    if args.dot_hex:
        dbgr = hex_to_bgr(args.dot_hex)
        dh, ds, dv = bgr_to_hsv_pixel(dbgr)
        print(f"Dot hex {args.dot_hex} -> BGR{dbgr} -> HSV({dh},{ds},{dv})")
        dot_bounds = bounds_from_hsv(dh, ds, dv, args.dot_hue_tol, 80, 40, single=True)
        print(f"Dot bounds: lo {dot_bounds[0]} hi {dot_bounds[1]}")

    frames = None
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f'ERROR: cannot read "{args.image}"', file=sys.stderr)
            sys.exit(1)
        frames = [img]
    elif args.refine:
        print(f"Hold dot in view - capturing {args.frames} frames...")
        frames = grab_camera_frames(args.source, args.frames)

    if frames is not None:
        res, count = refine_from_frames(frames, h, args.hue_tol)
        if res is None:
            print(f"WARNING: only {count} near-hue px - keeping hex-only bounds.")
        else:
            s_lo, v_lo, count = res
            print(f"Refined from {count} px: S floor {s_lo}, V floor {v_lo}")
            l1[1], l2[1] = s_lo, s_lo
            l1[2], l2[2] = v_lo, v_lo

    cfg = load_config(args.config)
    cfg["red_hsv"] = {"comment": f"Trained from card {args.hex} -> HSV({h},{s},{v}).",
                      "lower1": l1, "upper1": u1, "lower2": l2, "upper2": u2}
    if dot_bounds is not None:
        cfg["dot_hsv"] = {"comment": f"Trained from dot {args.dot_hex}.",
                          "lower": dot_bounds[0], "upper": dot_bounds[1]}
    save_config(cfg, args.config)
    print(f"Saved to {args.config}")

    if args.show:
        cap = cv2.VideoCapture(args.source)
        print("Preview - ESC to exit.")
        a1, b1 = np.array(l1), np.array(u1)
        a2, b2 = np.array(l2), np.array(u2)
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            hsv = cv2.cvtColor(cv2.GaussianBlur(fr, (7, 7), 0), cv2.COLOR_BGR2HSV)
            m = cv2.bitwise_or(cv2.inRange(hsv, a1, b1), cv2.inRange(hsv, a2, b2))
            cv2.imshow("train_red mask", m)
            cv2.imshow("train_red frame", fr)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

