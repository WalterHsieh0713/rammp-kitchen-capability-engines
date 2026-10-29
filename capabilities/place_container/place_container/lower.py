"""The arm side of placing, on self.arm: the impedance lowering with the contact hold, letting go,
the retreat, and leaving impedance. Ported from RAMMP-table's kinova_arm.py (2026-10-26).

The lowering is ONE execute_joint_trajectory goal in the driver's IMPEDANCE mode: hold the start
for `settle_s` (the arm sags a little when it goes compliant, and that must be over before the
descent, or it looks like motion), then the planned straight line down. Its monitor:
  - records every new /ee_state message (pose, twist, wrench, phase) for plot_descent.py;
  - contact (plan.VzContact) -> REPLACES the goal with a hold `hold_press` below the contact point
    on the planned line (LATEST_WINS). Not a cancel: after a cancel the compliant arm sprang 2.7 cm
    up (2026-10-08), and a setpoint AT the hand let it spring back a little.
The arm stays in impedance through the open and the retreat (switching back to position control
jerked it, 2026-10-19). Any other ending leaves impedance with the position setpoint ON the current
joints (exit_impedance), so a failed run never leaves the arm compliant in mid-air.
"""

import csv
import time

import numpy as np
from capability_engine.arm import IMPEDANCE, LATEST_WINS, POSITION, SUCCESSFUL
from rammp_arm import JOINTS, _unwrap_from

from .plan import PATH_STEP, RETREAT_STEP, Refused, VzContact, hold_point, plan_retreat

SLIDE_SPEED = 0.02  # m/s down
RETREAT_SPEED = 0.03  # m/s back
MAX_JOINT_SPEED = np.radians(30.0)  # rad/s, every move here is timed to stay under it
LOOSE_TOL_RAD = 0.5  # path/goal tolerance of the watched impedance goals: the compliant arm lags and
# is held back by the table; the monitor watches instead
HOLD_S = 0.5  # s, the hold goal's single point
Z_STILL_SPEED = 0.002  # m/s, |vz| under this = still
SETTLE_STILL_S = 0.5  # the settle must end with the hand still this long
EXIT_SETTLE_S = 2.0  # leaving impedance: wait at most this long for the arm to be still ...
EXIT_HOLD_S = 0.5  # ... then a position goal on the current joints, this long
OPEN_STALLED = 0.3  # the gripper must open past this, or the hand would drag the container
LOG_COLUMNS = ["t", "phase", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "fx", "fy", "fz", "tx", "ty", "tz"]


def timed(start, waypoints, min_step_s):
    """Times for the waypoints: each step at least min_step_s, and slow enough that no joint goes
    faster than MAX_JOINT_SPEED (x1.5 for the speed-up between waypoints)."""
    times, t, prev = [], 0.0, np.asarray(start, dtype=float)
    for q in waypoints:
        q = np.asarray(q, dtype=float)
        t += max(min_step_s, 1.5 * float(np.max(np.abs(q - prev))) / MAX_JOINT_SPEED)
        times.append(t)
        prev = q
    return times


def points(times, qs):
    """[(t, positions, velocities)] for execute_joint_trajectory; no velocities (the driver
    interpolates), as the RAMMP-table goals ran."""
    return [(float(t), [float(v) for v in q], []) for t, q in zip(times, qs, strict=True)]


class Recorder:
    """/ee_state rows while a goal runs (call poll() from its monitor), with the phase they were in."""

    def __init__(self):
        self.rows, self.phase, self._last = [], "settle", None

    def poll(self, arm):
        ee = arm.ee_state()
        key = (*ee.position, *ee.linear)
        if key != self._last:
            self._last = key
            self.rows.append([time.time(), self.phase, *ee.position, *ee.linear, *ee.angular, *ee.force, *ee.torque])
            return ee, True
        return ee, False

    def write(self, directory):
        if not self.rows:
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / time.strftime("descent_%Y%m%d_%H%M%S.csv")
        with open(path, "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(LOG_COLUMNS)
            out.writerows(self.rows)
        return path


def impedance_goal(arm, pts, gains, monitor=None):
    """One watched IMPEDANCE trajectory (LATEST_WINS: it replaces whatever holds the arm)."""
    return arm.execute_joint_trajectory(
        JOINTS,
        pts,
        path_tolerance=LOOSE_TOL_RAD,
        goal_tolerance=LOOSE_TOL_RAD,
        control_mode=IMPEDANCE,
        preemption=LATEST_WINS,
        gains=gains,
        monitor=monitor,
    )


def wait_still(arm, timeout_s, hold_s=0.2):
    """Wait until the flange's z velocity has been ~0 for hold_s (at most timeout_s)."""
    t0, since = time.monotonic(), None
    while time.monotonic() - t0 < timeout_s:
        still = abs(float(arm.ee_state().linear[2])) < Z_STILL_SPEED
        since = (since or time.monotonic()) if still else None
        if since is not None and time.monotonic() - since >= hold_s:
            return True
        time.sleep(0.02)
    return False


def exit_impedance(arm, log):
    """Back to POSITION control without a jump: in impedance the arm sits a few cm off its
    setpoint, so switching on that setpoint would pull it there. Wait until still, then a POSITION
    goal whose setpoint IS the current joints."""
    wait_still(arm, EXIT_SETTLE_S)
    q = arm.raw_joints()
    res = arm.execute_joint_trajectory(
        JOINTS, points([EXIT_HOLD_S], [q]), control_mode=POSITION, preemption=LATEST_WINS
    )
    log(
        "impedance off: position setpoint = the current joints"
        + ("" if res.error_code == SUCCESSFUL else f" ({res.error_string})")
    )


def lower(arm, model, descent, cfg, rec, log):
    """Lower along `descent` (plan.plan_descent's (zs, qs)) until contact, then hold. Returns the
    hold's joints (where the retreat starts). Raises Refused (nothing let go) if it didn't settle,
    nothing stopped it, or it stopped well above the table; the arm is then out of impedance."""
    _, qs = descent
    line = _unwrap_from(arm.raw_joints(), qs)  # the short way round from the measured joints
    line_z = [float(model.fk(q, "flange")[2, 3]) for q in qs]
    settle_s = float(cfg["settle_s"])
    times = [settle_s] + [settle_s + t for t in timed(line[0], line[1:], PATH_STEP / SLIDE_SPEED)]
    down_s = times[-1] - settle_s
    contact = VzContact(stop=cfg["vz_stop"])
    state = dict(t0=None, still_since=None, lag=None, hold=None, why=None, z=None)

    def monitor():
        ee, new = rec.poll(arm)
        state["t0"] = state["t0"] or time.monotonic()  # the goal's clock, from its first poll
        t, z, vz = time.monotonic() - state["t0"], float(ee.position[2]), float(ee.linear[2])
        if t < settle_s:  # 1. settling in impedance, holding the start
            if abs(vz) < Z_STILL_SPEED:
                state["still_since"] = state["still_since"] or t
            else:
                state["still_since"] = None
            if t > settle_s - 0.05:
                still = state["still_since"] is not None and t - state["still_since"] >= SETTLE_STILL_S
                if not still:
                    state["why"] = f"not still after {t:.1f} s in impedance (vz {vz * 100:+.2f} cm/s) -- not lowering"
                    return points([HOLD_S], [line[0]])
                state["lag"] = z - line_z[0]
                rec.phase = "descend"
            return None
        if state["lag"] is None:  # the settle's last poll was missed: measure the lag now
            state["lag"], rec.phase = z - line_z[0], "descend"
        if new and contact.update(vz) and (t - settle_s) < 0.99 * down_s:  # 2. contact
            q = hold_point(line, line_z, z - state["lag"] - cfg["hold_press"])
            state["hold"] = np.asarray(arm.raw_joints() if q is None else q, dtype=float)
            state["z"], rec.phase = z, "contact"
            log(
                f"contact at flange z={z:.3f}: vz {vz * 100:+.2f} cm/s, {(t - settle_s) / down_s * 100:.0f}% "
                f"down -> holding {cfg['hold_press'] * 1000:.0f} mm below it (impedance)"
            )
            return points([HOLD_S], [state["hold"]])
        return None

    log(
        f"lowering: impedance ({cfg['impedance_gains']}), settle {settle_s:.1f} s, then {SLIDE_SPEED * 100:.0f} cm/s "
        f"down to tool z={float(descent[0][-1]):.3f}; contact = vz back up to {cfg['vz_stop'] * 100:+.1f} cm/s"
    )
    res = impedance_goal(arm, points([0.0] + times, [line[0]] + line), cfg["impedance_gains"], monitor)
    if state["hold"] is None:
        why = state["why"] or (
            f"went down {down_s * SLIDE_SPEED * 100:.0f} cm without stopping on anything ({res.error_string or 'done'})"
        )
        exit_impedance(arm, log)
        raise Refused(f"STOPPED: {why} -- nothing let go")
    wait_still(arm, 3.0)
    z_tool = float(model.fk(arm.raw_joints())[2, 3])
    rest = z_tool - cfg["table_z"] - cfg["bowl_depth"]
    if rest > cfg["early_stop"]:
        raise Refused(
            f"STOPPED: the hand stopped at tool z={z_tool:.3f}, {rest * 100:.0f} cm above where the bottom meets "
            "the table -- it hit something else; NOT letting go (holding in impedance)"
        )
    log(f"contact at tool z={z_tool:.3f}")
    return state["hold"]


def let_go(arm, rec, log):
    """Open the gripper; Refused (hand NOT backed out) if it stays closed on the container."""
    rec.phase = "open"
    arm.gripper(0.0)
    pos = float(arm.gripper_state().position)
    log(f"gripper open: {pos:.2f} (0 = fully open)")
    if pos > OPEN_STALLED:
        raise Refused(f"STOPPED, hand NOT backed out: the gripper only opened to {pos:.2f}. Open it, then `retreat`")


def back_out(arm, model, q_start, cfg, rec, log, release=0.0):
    """Pull the hand straight back along plan.plan_retreat's line, in impedance (it replaces the
    hold), at RETREAT_SPEED."""
    line = _unwrap_from(arm.raw_joints(), plan_retreat(model, q_start, release=release, log=log))
    rec.phase = "retreat"
    log(
        f"retreat: {(len(line) - 1) * RETREAT_STEP * 100:.0f} cm straight back "
        f"(impedance, {RETREAT_SPEED * 100:.0f} cm/s)"
    )
    times = timed(line[0], line[1:], RETREAT_STEP / RETREAT_SPEED)

    def record_only():
        rec.poll(arm)  # None: never stops the retreat

    res = impedance_goal(arm, points([0.0] + times, line), cfg["impedance_gains"], monitor=record_only)
    if res.error_code != SUCCESSFUL:
        raise Refused(f"STOPPED: the retreat did not complete ({res.error_string})")
    wait_still(arm, 3.0)
    rec.phase = "done"


def joint_moves(arm, model, moves, cfg, log, tol_deg=3.0):
    """Each (name, q) as its own impedance joint move (reset): timed under MAX_JOINT_SPEED, and
    checked to end within tol_deg."""
    for name, q in moves:
        T = model.fk(q)
        log(f"-> {name}: hand to x={T[0, 3]:.3f} y={T[1, 3]:.3f} z={T[2, 3]:.3f} (impedance)")
        now = arm.raw_joints()
        (target,) = _unwrap_from(now, [q])
        t = max(1.0, timed(now, [target], 0.0)[0])
        res = impedance_goal(arm, points([0.0, t], [now, target]), cfg["impedance_gains"])
        wait_still(arm, 3.0)
        off = float(np.max(np.abs(np.degrees(np.asarray(arm.raw_joints()) - target))))
        if res.error_code != SUCCESSFUL or off > tol_deg:
            raise Refused(f"STOPPED: {name} ended {off:.1f} deg off ({res.error_string or 'done'})")
