"""Replay a teleop keypoint demo relative to the microwave door (push-open, push-close, ...). Dry run unless --execute.

    python3 -m microwave.demo_replay encode demos/push_open_2026-10-07.json   # -> demos/push_open_2026-10-07.door.json
    python3 -m microwave.demo_replay run demos/push_open_2026-10-07.door.json [--trust-demo 3] [--execute]

A demo is recorded with `tools/kp.sh` (one keypoint per pose, arm teleoped in between) plus a
copy of the door file at that time (`<demo>_door.json`). `encode` turns it into DOOR coordinates,
so a replay adapts to where today's microwave is, how it's turned, how big it is and how high:

* horizontal, in the closed door's frame at the detected hinge: `a` = along the door (hinge ->
  handle), as a fraction of the hinge-to-handle distance (scales with the door's size); `b` =
  along the door's outward normal (toward the arm), in metres (clearance doesn't scale);
* height: metres above the door top for points that were above it, else metres from the grasp height;
* hand orientation relative to the door's facing.

`run` maps it onto today's door file (microwave_common.DOOR_FILE), checks the hand is at the demo's
first keypoint, sim-gates every leg (IK, joint jump, J4/J6, wrap, >= DEMO_MIN_CLEAR to the door
model; legs from --trust-demo on against the microwave body only -- the 10-07 door model put the
user's REAL push-open poses 2-7 cm inside the door), then (--execute) does the demo's gripper action
and follows the keypoints as one smooth trajectory.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rammp_arm
from scipy.spatial.transform import Rotation as R

from . import microwave_common as mc
from .microwave_common import (
    DoorFrame,
    drop,
    make_sim,
    plan_cartesian,
    run_cartesian_trajectory,
    say,
)

OVER_TOP_MIN = 0.08  # points above the door top stay at least this far above today's top
START_TOL = 0.06  # the hand must be within this of the demo's first keypoint
DEMO_MIN_CLEAR = 0.02  # to the door model, on legs before --trust-demo
LIFT_OFF_MIN_CLEAR = 0.005  # first leg of a demo that starts holding the handle: the fingers straddle the bar, i.e.
# start next to the door face (as the old door_push.RETREAT_OUT_MIN_CLEAR)


def door_frame(door):
    """(hinge xy, e = hinge->handle along the closed door, n = outward normal, r = hinge-handle distance, yaw of n)."""
    h = np.asarray(door["hinge"][:2], float)
    e = np.asarray(door["closed_handle"][:2], float) - h
    r = float(np.linalg.norm(e))
    e /= r
    n = np.asarray(door["closed_normal"][:2], float)
    n -= e * float(n @ e)
    n /= np.linalg.norm(n)
    return h, e, n, r, float(np.arctan2(n[1], n[0]))


def encode(demo_path, adjust=None, start=0):
    """Keypoint demo (+ its _door.json) -> door-coordinate demo. `adjust`: {index: (da metres, db metres)}
    (indices of the RECORDED keypoints); `start`: drop the recorded keypoints before this one."""
    demo_path = Path(demo_path)
    demo = json.loads(demo_path.read_text())
    door = json.loads(demo_path.with_name(demo_path.stem + "_door.json").read_text())
    h, e, n, r, yaw = door_frame(door)
    top, zg = float(door["door_z"][1]), float(door["closed_grasp_pos"][2])
    out = []
    for i, k in enumerate(demo["keypoints"]):
        if i < start:
            continue
        p = np.asarray(k["tool"], float)
        d = p[:2] - h
        da, db = (adjust or {}).get(i, (0.0, 0.0))
        rel_q = (R.from_euler("z", -yaw) * R.from_quat(k["quat"])).as_quat()
        out.append(
            {
                "label": k["label"],
                "a": float((d @ e + da) / r),
                "b": float(d @ n + db),
                "z_ref": "top" if p[2] > top else "grasp",
                "dz": float(p[2] - (top if p[2] > top else zg)),
                "quat_door": rel_q.tolist(),
                "gripper": k["gripper"],
            }
        )
    return {
        "source": demo_path.name,
        "door_open_deg": door.get("door_open_deg"),
        "hinge_to_handle_m": r,
        "keypoints": out,
        "adjust": {str(i): v for i, v in (adjust or {}).items()},
        "start": start,
        "notes": demo.get("notes", []),
    }


def decode(enc, door):
    """Door-coordinate demo -> [(pos, quat)] on today's door."""
    h, e, n, r, yaw = door_frame(door)
    top, zg = float(door["door_z"][1]), float(door["closed_grasp_pos"][2])
    pts = []
    for k in enc["keypoints"]:
        xy = h + e * k["a"] * r + n * k["b"]
        z = (top + max(k["dz"], OVER_TOP_MIN)) if k["z_ref"] == "top" else zg + k["dz"]
        pts.append((np.r_[xy, z], (R.from_euler("z", yaw) * R.from_quat(k["quat_door"])).as_quat()))
    return pts


def plan(scene, rb, door, st, kps, trust_from, lift_off=False):
    """Sim-gated joint path through `kps` from the arm's state `st`: one list of joints per leg, or None."""
    df = DoorFrame(door, door.get("open_sign", -1.0))
    q, legs = np.asarray(st["position"], float), []
    bodies = df.bodies(scene, rb, float(door.get("door_open_deg", 50.0)))
    try:
        pts = [(np.asarray(st["ee_pos"][:3]), np.asarray(st["ee_pos"][3:7]))] + kps
        for i in range(len(pts) - 1):
            trusted = trust_from is not None and i + 1 >= trust_from
            check = [b for b in bodies if b[1] == "microwave body"] if trusted else bodies
            clear = LIFT_OFF_MIN_CLEAR if (i == 0 and lift_off) else DEMO_MIN_CLEAR
            leg = plan_cartesian(
                scene,
                rb,
                pts[i][0],
                pts[i][1],
                pts[i + 1][0],
                pts[i + 1][1],
                q,
                f"{'trusted' if trusted else 'leg'} {i + 1}",
                check,
                min_clear=clear,
            )
            if leg is None:
                return None
            legs.append(leg)
            q = leg[-1]
    finally:
        drop(rb, bodies)
    return legs


def run(enc_path, execute, trust_from, ai=None):
    """Replay one encoded demo. `ai`: the arm to use (open_door chains steps); else one is made here.
    Returns True when it reached the last keypoint (or the dry run passed)."""
    enc = json.loads(Path(enc_path).read_text())
    door = json.loads(mc.DOOR_FILE.read_text())
    pts = decode(enc, door)
    _, _, _, r, _ = door_frame(door)
    say(
        f"demo {enc['source']}: {len(pts)} keypoints; hinge-handle {enc['hinge_to_handle_m'] * 100:.1f} cm then, "
        f"{r * 100:.1f} cm now; door file says ~{door.get('door_open_deg', 0):.0f} deg open"
    )
    for k, (p, _) in zip(enc["keypoints"], pts, strict=False):
        say(f"  {k['label'][:30]:30s} -> {np.round(p, 3)}")
    return replay(pts, [k["gripper"] for k in enc["keypoints"]], door, execute, trust_from, ai)


def replay(pts, grippers, door, execute, trust_from, ai=None):
    """`run` from the mapped keypoints on: `pts` [(pos, quat)] (the first = where the hand must be now), `grippers` the
    gripper reading at each, `door` the door dict the legs are checked against. Also what the fridge push-open runs
    (fridge_push.py: its keypoints are computed for today's door, not decoded from a recorded demo)."""
    own = ai is None
    ai = ai or rammp_arm.RammpArm()
    try:
        st = ai.get_state()
        off = float(np.linalg.norm(np.asarray(st["ee_pos"][:3]) - pts[0][0]))
        say(
            f"hand now {np.round(st['ee_pos'][:3], 3)}, {off * 100:.1f} cm from the demo's start; "
            f"gripper {st['gripper_pos']:.2f}"
        )
        if off > START_TOL:
            say(f"not at the demo's start (> {START_TOL * 100:.0f} cm) -- refusing.")
            return False
        scene, rb = make_sim()
        lift_off = grippers[0] > 0.2 and grippers[1] < 0.2
        legs = plan(scene, rb, door, st, pts[1:], trust_from, lift_off)
        if legs is None:
            say("a leg fails a gate -- refusing, arm untouched.")
            return False
        say(f"all {len(legs)} legs pass ({sum(len(leg) for leg in legs)} steps).")
        if not execute:
            say("\nDRY RUN -- nothing commanded.")
            return True
        if lift_off:  # the demo let go of the handle first
            say("opening the gripper ...")
            ai.gripper(0.0)
            time.sleep(0.5)
        # The push-further-open CAN'T HANDLE MORE SPEED: always run it at the contact speed
        # (rammp_arm.CONTACT_*_SPEED = 0.5 rad/s / 0.09 m/s), never the faster free-air speed. 10-24: at
        # 0.65/0.12 and 0.85/0.16 its last leg (keypoint 4 -> 5: the hand pushing into the door, which pushes
        # back, while turning 68 deg) aborted PATH_TOLERANCE_VIOLATED ~12.5 cm short both times; it ran clean
        # three times at 0.5 / 0.09.
        ok, err, _ = run_cartesian_trajectory(
            ai, [(list(p), list(q)) for p, q in pts[1:]], speed_scale=rammp_arm.contact_speed_scale()
        )
        say(
            f"{'DONE' if ok else 'STOPPED SHORT'}: hand {np.round(ai.get_state()['ee_pos'][:3], 3)} "
            f"({err * 100:.1f} cm from the last keypoint)"
        )
        return bool(ok)
    finally:
        if own:
            ai.close()


def main():
    a = argparse.ArgumentParser()
    sub = a.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("demo")
    e.add_argument(
        "--adjust",
        default="",
        help="'i:da:db,...' metres added at recorded keypoint i (along the door "
        "hinge->handle, off its face toward the arm)",
    )
    e.add_argument("--start", type=int, default=0, help="first recorded keypoint to keep")
    r_ = sub.add_parser("run")
    r_.add_argument("encoded")
    r_.add_argument("--execute", action="store_true")
    r_.add_argument("--trust-demo", type=int, default=None, metavar="LEG")
    args = a.parse_args()
    if args.cmd == "encode":
        adj = {int(i): (float(da), float(db)) for i, da, db in (t.split(":") for t in args.adjust.split(",") if t)}
        enc = encode(args.demo, adj, args.start)
        out = Path(args.demo).with_suffix(".door.json")
        out.write_text(json.dumps(enc, indent=1))
        say(f"wrote {out}")
        for k in enc["keypoints"]:
            say(
                f"  {k['label'][:30]:30s} a {k['a']:+.3f} x r   b {k['b'] * 100:+6.1f} cm   "
                f"z {k['dz'] * 100:+6.1f} cm from {k['z_ref']}"
            )
    else:
        mc.configure_paths(mc.DATA_DIR)
        sys.exit(0 if run(args.encoded, args.execute, args.trust_demo) else 1)


if __name__ == "__main__":
    main()
