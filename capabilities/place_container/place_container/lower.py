"""The arm side of placing, on self.arm as it is: the impedance lowering with the contact hold, letting
go, the retreat, and leaving impedance. Ported from RAMMP-table's kinova_arm.py (2026-10-26).

1. settle: an IMPEDANCE goal holding the line's start for `settle_s` (execute_joint_trajectory);
   the hand must then be still -- the arm sags a little when it goes compliant, and that must be
   over before the descent, or it looks like motion.
2. descent: the planned straight line down, self.arm.move_joints in impedance. Its monitor records
   every new /ee_state message (pose, twist, wrench, phase) for plot_descent.py and stops on
   contact (plan.VzContact): move_joints CANCELS the goal, and the hold goes out right after.
   move_joints times the whole line minimum-jerk (slow at both ends), so contact only counts while
   the commanded speed is up (plan.contact_gate); the speed is chosen so the hand passes the
   marked table at ~lower_speed (plan.descent_speed_scale).
   RAMMP-table replaced the goal with the hold instead of cancelling it (a cancel made the compliant
   arm spring 2.7 cm up on the HERL chair, 2026-10-08, SOFT gains); whether the flatbot does so is
   the first thing to look at in the recording (z right after "contact").
3. hold `hold_press` below the contact point on the planned line, while the gripper opens.
4. retreat straight back, move_joints in impedance.
One set of impedance gains (`gains`) for all of it, so the stiffness never changes mid-place. Any
ending before contact leaves impedance with the position setpoint ON the current joints
(exit_impedance), so a failed run never leaves the arm compliant in mid-air.
"""

import csv
import time

import numpy as np
import rammp_arm
from capability_engine.arm import IMPEDANCE, LATEST_WINS, POSITION, SUCCESSFUL
from rammp_arm import JOINTS, _unwrap_from

from .plan import Refused, VzContact, contact_gate, descent_speed_scale, hold_point, plan_retreat

LOOSE_TOL_RAD = 0.5  # path/goal tolerance of the impedance moves: the compliant arm lags and is held back
# by the table; the monitor watches instead
HOLD_S = 0.5  # s, the settle's and hold's single point
Z_STILL_SPEED = 0.002  # m/s, |vz| under this = still
SETTLE_STILL_S = 0.5  # after the settle the hand must be still this long ...
SETTLE_WAIT_S = 5.0  # ... within this long
EXIT_SETTLE_S = 2.0  # leaving impedance: wait at most this long for the arm to be still ...
EXIT_HOLD_S = 0.5  # ... then a position goal on the current joints, this long
OPEN_STALLED = 0.3  # the gripper must open past this, or the hand would drag the container ...
OPEN_WAIT_S = 3.0  # ... within this long of the open command (sent twice at most)
LOG_COLUMNS = ["t", "phase", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "fx", "fy", "fz", "tx", "ty", "tz"]


def points(times, qs):
    """[(t, positions, velocities)] for execute_joint_trajectory; no velocities (the driver
    interpolates), as the RAMMP-table goals ran."""
    return [(float(t), [float(v) for v in q], []) for t, q in zip(times, qs, strict=True)]


def compliant(gains):
    """move_joints' `impedance` dict for `gains` (kq, zeta, torque_limit)."""
    return {**gains, "path_tol": LOOSE_TOL_RAD}


class Recorder:
    """/ee_state rows (poll() from a monitor or a wait), with the phase they were in."""

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


def hold_goal(arm, q, gains, hold_s=HOLD_S):
    """An IMPEDANCE goal holding joints q (LATEST_WINS: it replaces whatever holds the arm)."""
    return arm.execute_joint_trajectory(
        JOINTS,
        points([hold_s], [q]),
        path_tolerance=LOOSE_TOL_RAD,
        goal_tolerance=LOOSE_TOL_RAD,
        control_mode=IMPEDANCE,
        preemption=LATEST_WINS,
        gains=gains,
    )


def wait_still(arm, timeout_s, hold_s=0.2, rec=None):
    """Wait until the flange's z velocity has been ~0 for hold_s (at most timeout_s)."""
    t0, since = time.monotonic(), None
    while time.monotonic() - t0 < timeout_s:
        ee = rec.poll(arm)[0] if rec is not None else arm.ee_state()
        still = abs(float(ee.linear[2])) < Z_STILL_SPEED
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
    zs, qs = descent
    gains = cfg["gains"]
    line = _unwrap_from(arm.raw_joints(), qs)  # the short way round from the measured joints
    line_z = [float(model.fk(q, "flange")[2, 3]) for q in qs]

    def fail(why):
        exit_impedance(arm, log)
        return Refused(f"STOPPED: {why} -- nothing let go")

    # 1. settle in impedance on the line's start
    rec.phase = "settle"
    log(f"impedance on (kq {gains['kq'][0]:g}..), holding the start {cfg['settle_s']:.1f} s ...")
    res = hold_goal(arm, line[0], gains, hold_s=cfg["settle_s"])
    if res.error_code != SUCCESSFUL:
        raise fail(f"the impedance hold at the start ended {res.error_string}")
    if not wait_still(arm, SETTLE_WAIT_S, SETTLE_STILL_S, rec):
        raise fail(f"the hand is not still in impedance after {cfg['settle_s'] + SETTLE_WAIT_S:.0f} s -- not lowering")
    lag = float(arm.ee_state().position[2]) - line_z[0]
    log(f"settled, hand {lag * 100:+.1f} cm {'above' if lag >= 0 else 'below'} the planned start")

    # 2. the line down, stopped on contact
    lengths = np.abs(np.diff(zs))
    meet = (zs[0] - (cfg["table_z"] + cfg["bowl_depth"])) / (zs[0] - zs[-1])  # where the table should be, 0..1
    scale = descent_speed_scale(meet, cfg["lower_speed"], rammp_arm.CART_SPEED)
    samples = rammp_arm.joint_trajectory(arm.raw_joints(), line[1:], lengths, scale)
    if samples is None:
        raise fail("the line down is too short to move")
    duration, length = float(samples[0][-1]), float(np.sum(lengths))
    contact = VzContact(stop=cfg["vz_stop"])
    state = dict(t0=None, z=None)

    def monitor():
        ee, new = rec.poll(arm)
        state["t0"] = state["t0"] or time.monotonic()  # the goal's clock, from its first poll
        u = (time.monotonic() - state["t0"]) / duration
        hit = new and contact.update(float(ee.linear[2]))
        if hit and contact_gate(u, length, duration, cfg["lower_speed"]):
            state["z"] = float(ee.position[2])
            rec.phase = "contact"
            log(
                f"contact at flange z={state['z']:.3f}: vz {ee.linear[2] * 100:+.2f} cm/s, "
                f"{min(u, 1.0) * 100:.0f}% of the move -> stop, hold {cfg['hold_press'] * 1000:.0f} mm below it"
            )
            return True
        return False

    rec.phase = "descend"
    log(
        f"lowering: {length * 100:.0f} cm in {duration:.0f} s (impedance), ~{cfg['lower_speed'] * 100:.0f} cm/s "
        f"where the table should be ({meet * 100:.0f}% down); contact = vz back up to {cfg['vz_stop'] * 100:+.1f} cm/s"
    )
    ok = arm.move_joints(line[1:], cart_lengths=lengths, speed_scale=scale, impedance=compliant(gains), monitor=monitor)
    if state["z"] is None:
        raise fail(
            f"went down {length * 100:.0f} cm without stopping on anything"
            if ok
            else f"the descent failed ({arm.last_error})"
        )

    # 3. hold a little below the contact point
    q = hold_point(line, line_z, state["z"] - lag - cfg["hold_press"])
    hold_q = np.asarray(arm.raw_joints() if q is None else q, dtype=float)
    hold_goal(arm, hold_q, gains)
    wait_still(arm, 3.0, rec=rec)
    z_tool = float(model.fk(arm.raw_joints())[2, 3])
    rest = z_tool - cfg["table_z"] - cfg["bowl_depth"]
    if rest > cfg["early_stop"]:
        raise Refused(
            f"STOPPED: the hand stopped at tool z={z_tool:.3f}, {rest * 100:.0f} cm above where the bottom meets "
            "the table -- it hit something else; NOT letting go (holding in impedance)"
        )
    log(f"contact at tool z={z_tool:.3f}")
    return hold_q


def let_go(arm, rec, log):
    """Open the gripper; Refused (hand NOT backed out) if it stays closed on the container."""
    rec.phase = "open"
    pos = 1.0
    for attempt in (1, 2):  # the open command once more if the fingers haven't moved
        arm.gripper(0.0)
        t0 = time.monotonic()
        while time.monotonic() - t0 < OPEN_WAIT_S:  # it may start late (flatbot, 2026-10-29: 0.95 right after)
            pos = float(arm.gripper_state().position)
            if pos <= OPEN_STALLED:
                break
            time.sleep(0.05)
        if pos <= OPEN_STALLED:
            break
        log(
            f"gripper still at {pos:.2f} after {OPEN_WAIT_S:.0f} s" + (" -- sending open again" if attempt == 1 else "")
        )
    log(f"gripper open: {pos:.2f} (0 = fully open)")
    if pos > OPEN_STALLED:
        raise Refused(f"STOPPED, hand NOT backed out: the gripper only opened to {pos:.2f}. Open it, then `retreat`")


def back_out(arm, model, q_start, cfg, rec, log, release=0.0):
    """Pull the hand straight back along plan.plan_retreat's line, in impedance, at most
    `retreat_speed` (the line is timed minimum-jerk)."""
    line = plan_retreat(model, q_start, release=release, log=log)
    lengths = [
        float(np.linalg.norm(model.fk(b)[:3, 3] - model.fk(a)[:3, 3])) for a, b in zip(line[:-1], line[1:], strict=True)
    ]
    rec.phase = "retreat"
    log(f"retreat: {sum(lengths) * 100:.0f} cm straight back (impedance, up to {cfg['retreat_speed'] * 100:.0f} cm/s)")

    def record_only():
        rec.poll(arm)
        return False  # never stops the retreat

    ok = arm.move_joints(
        line[1:],
        cart_lengths=lengths,
        speed_scale=cfg["retreat_speed"] / rammp_arm.CART_SPEED,
        impedance=compliant(cfg["gains"]),
        monitor=record_only,
    )
    if not ok:
        raise Refused(f"STOPPED: the retreat did not complete ({arm.last_error})")
    wait_still(arm, 3.0, rec=rec)
    rec.phase = "done"


def joint_moves(arm, model, moves, cfg, log, tol_deg=3.0):
    """Each (name, q) as its own impedance joint move (reset), at `reset_speed_scale` of self.arm's
    joint speed, checked to end within tol_deg."""
    for name, q in moves:
        T = model.fk(q)
        log(f"-> {name}: hand to x={T[0, 3]:.3f} y={T[1, 3]:.3f} z={T[2, 3]:.3f} (impedance)")
        ok = arm.move_joints([q], speed_scale=cfg["reset_speed_scale"], impedance=compliant(cfg["gains"]))
        wait_still(arm, 3.0)
        (target,) = _unwrap_from(arm.raw_joints(), [q])
        off = float(np.max(np.abs(np.degrees(np.asarray(arm.raw_joints()) - target))))
        if not ok or off > tol_deg:
            raise Refused(f"STOPPED: {name} ended {off:.1f} deg off ({arm.last_error or 'done'})")
