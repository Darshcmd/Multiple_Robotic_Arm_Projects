# Red-Square + Dot Follower (Object-Detection)

Tracks a red square card with a dark dot in the center. Default setup: the
**laptop webcam (camera 0)** on a tripod/desk (fixed camera, absolute
pixel->angle mapping). Eye-in-hand mode (webcam rides the arm, IBVS servoing
to frame center) is still available with `--mode eye-in-hand --source 1`.
Built on the same ideas as
`Gesture-Controlled-5-DOF-Robotic-Arm-using-Computer-Vision`:

- Gaussian blur -> dual-range HSV mask (red wraps at hue 0) -> erode/dilate
- Stage 1: best square blob wins (squareness = aspect x fill of the ROTATED
  min-area rect — rotation-robust, works tilted up to ~45 deg). Cards cropped
  by the frame edge are rejected by a border-touch gate.
- Stage 2: dark dot found *inside* the card ROI (single HSV range, no wrap);
  the dot centroid is the servo target, card center is the fallback
- Two control modes:
  - `static` (default): absolute pixel->angle map from the fixed laptop cam
  - `eye-in-hand`: each frame nudges joints to shrink the dot-to-frame-center
    error (classic IBVS, one small step per frame)
- EMA smoothing on pixels AND joint angles, deadbands, slew-rate limits
- Framed serial `0xAA + arm1,arm2,tilt,base,claw`, send-on-change only
- Vision-only fallback when no Arduino is found

## Safety: arm can never over-extend

1. `red_config.json -> arm_limits` are NARROWER than the firmware 0-180 range
   (base 20-160, shoulder 50-160, elbow 10-154, wrist 10-150).
2. Every command is clamped after the servo step (belt and braces).
3. `SlewLimiter` caps degrees-per-frame (default 2-3 deg) + EMA 0.25, so
   corrections become slow glides instead of jerks.
4. `servo.deadband` (0.03): tiny errors near center are ignored — no hunting.
5. On marker loss the arm HOLDS its last pose (no snap-back).

## Train your card + dot hex colours

```bash
cd Object-Detection
../Gesture-Controlled-5-DOF-Robotic-Arm-using-Computer-Vision/venv/bin/python train_red.py --hex "#E8908A" --dot-hex "#2F4A55"
# live refinement under your lighting (laptop camera is 0, arm cam is 1):
../Gesture-Controlled-5-DOF-Robotic-Arm-using-Computer-Vision/venv/bin/python train_red.py --hex "#E8908A" --refine --frames 60 --source 0 --show
# from a photo:
../Gesture-Controlled-5-DOF-Robotic-Arm-using-Computer-Vision/venv/bin/python train_red.py --hex "#E8908A" --image ./card.jpg
```

Card hex builds the dual red range; `--dot-hex` builds the single dot range.
Both write into `red_config.json` (`red_hsv` + `dot_hsv`). Re-run whenever
lighting or marker material changes. Current bounds are tuned for the laptop
camera: card H~165 S~127 V~130 (S floor 70 cuts desk-brown leak), dot H~100
S~156 V~142 (V floor 40 for darker rooms).

## Run the follower (default: laptop camera, static mapping)

```bash
cd Object-Detection
# NORMAL RUN - drives the arm (Arduino auto-detected + hotplug auto-reconnect):
../Gesture-Controlled-5-DOF-Robotic-Arm-using-Computer-Vision/venv/bin/python red_dot_track.py --show-mask
# keep the dot centred (eye-in-hand servo; arm webcam):
../Gesture-Controlled-5-DOF-Robotic-Arm-using-Computer-Vision/venv/bin/python red_dot_track.py --mode eye-in-hand --source 1 --show-mask
# vision-only test (arm explicitly DISABLED - arm will NOT move):
../Gesture-Controlled-5-DOF-Robotic-Arm-using-Computer-Vision/venv/bin/python red_dot_track.py --no-arm --show-mask
```

Bottom-left badge shows the link truth: `ARM: LIVE (port) | ACK xN` (green =
packets sent AND confirmed by the firmware), `NO ACK (N pkts)` (red = wire/firmware
problem) or `ARM: SEARCHING - PLUG IN USB` (orange = board not present; it will
attach by itself within ~2 s of being plugged in). Terminal mirrors every step.

Hold the card in view: **only the base, shoulder and elbow track the card** -
the wrist/tilt motor stays parked at `mapping.tilt_fixed` (60 deg), so the
camera and claw keep their orientation. Move the card slowly at first.

Keys: `SPACE` pause/resume, `E` claw, `Q` quit (terminal or on-video buttons).

## Tune servoing / smoothness / limits

All in `red_config.json`, no code edits needed:

- `servo.gain_base/arm1/arm2`: deg per unit error. If the arm runs AWAY from
  the card, flip the sign of the offending gain.
- `mapping.tilt_fixed` (60) + `servo.gain_tilt` (0): the wrist is parked by
  default - only base/shoulder/elbow move. Set `gain_tilt` > 0 to let the
  wrist nudge with vertical error.
- `servo.deadband` (0.03): raise to ignore more center jitter.
- `servo.size_hold` (false): set true to also hold card apparent size
  (approach/back-off). Tune `target_size` from the `nx ny` HUD readout.
- `smoothing.angle_ema_alpha` (0.25): lower = smoother but laggier.
- `smoothing.max_step_*`: lower = slower, safer glides.
- `dot_hsv` / `red_hsv`: retrain via `train_red.py`, or hand-tweak bounds.
- `tracking.min_dot_area` (300): raise if specks false-trigger the dot.
- `arm_limits.*`: tighten further for a smaller/safer work envelope.

