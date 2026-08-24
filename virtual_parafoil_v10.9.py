"""
VIRTUAL PARAFOIL V10.9
REACHABILITY-BASED WIND-AWARE GUIDANCE
Ready-to-run standalone simulation.

Dependencies: Python 3.x, numpy
"""

import math
import random
import statistics
from dataclasses import dataclass
from typing import List, Dict, Tuple

try:
    import numpy as np
except ImportError:
    raise SystemExit("This program requires numpy. Install with: pip install numpy")


# =========================
# CONFIGURATION
# =========================
AREA = 0.96
MASS = 1.0
CL = 0.40
CD = 0.25

RHO = 1.225
G = 9.81

# Parafoil aerodynamic model
LIFT_TO_DRAG = CL / CD
AIR_SPEED = 7.013221573570879
HORIZONTAL_AIR_SPEED = AIR_SPEED / math.sqrt(1.0 + (1.0 / LIFT_TO_DRAG) ** 2)
VERTICAL_DESCENT = 3.717
GLIDE_RATIO = HORIZONTAL_AIR_SPEED / VERTICAL_DESCENT

TARGET_X = 500.0
TARGET_Y = 200.0
TOLERANCE = 20.0

INITIAL_X = 0.0
INITIAL_Y = 0.0
INITIAL_ALTITUDE = 600.0

REF_WIND_SPEED = 3.0
REF_WIND_DIR_DEG = 0.0

GNSS_POS_NOISE = 3.0
GNSS_VEL_NOISE = 0.30
BARO_NOISE = 2.0
IMU_HEADING_NOISE_DEG = 2.0
IMU_TURN_NOISE_DEG_S = 0.5
AIRSPEED_NOISE = 0.20

GNSS_RATE = 5.0
BARO_RATE = 10.0
IMU_RATE = 50.0
AIRSPEED_RATE = 20.0

GUIDANCE_INTERVAL = 2.0
NUM_COMMANDS = 21
MAX_TURN_RATE_DEG_S = 15.0
MAX_STEERING = 1.0

WIND_ALPHA = 0.08
MAX_WIND_ESTIMATE = 12.0
WIND_UNCERTAINTY_FLOOR = 0.25

# V10.9 candidate trajectory model
CANDIDATE_SIM_DT = 0.5
MAX_PREDICTION_TIME = 60.0

# Cost weights
W_POSITION = 1.00
W_CROSS_TRACK = 0.30
W_HEADING = 12.0
W_STEERING = 1.5
W_OVERSHOOT = 2.0
W_WIND_UNCERTAINTY = 0.50

# Steering dynamics
STEERING_RATE_PER_SEC = 0.35
STEERING_DEADBAND = 0.015
TARGET_CAPTURE_RADIUS = 20.0

# Monte Carlo
MC_RUNS = 100
MC_SEED = 109

DIRECTION_STUDY = list(range(0, 360, 45))
SPEED_STUDY = [0, 1, 2, 3, 4, 5, 6, 7]


# =========================
# UTILITIES
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_angle_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def angle_deg(x, y):
    return math.degrees(math.atan2(y, x)) % 360.0


def wind_components(speed, direction_deg):
    r = math.radians(direction_deg)
    return speed * math.cos(r), speed * math.sin(r)


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def fmt_status(error):
    return "PASS" if error <= TOLERANCE else "OUTSIDE TOLERANCE"


def adaptive_horizon(altitude):
    if altitude > 400:
        return 20.0
    if altitude > 200:
        return 15.0
    if altitude > 100:
        return 10.0
    return 5.0


# =========================
# SENSOR HEALTH
# =========================
@dataclass
class SensorHealth:
    gnss: float = 1.0
    imu: float = 1.0
    baro: float = 1.0
    airspeed: float = 1.0

    def recover(self, name, amount=0.02):
        value = getattr(self, name)
        setattr(self, name, clamp(value + amount, 0.0, 1.0))

    def degrade(self, name, amount=0.06):
        value = getattr(self, name)
        setattr(self, name, clamp(value - amount, 0.0, 1.0))


# =========================
# WIND ESTIMATOR
# =========================
class WindEstimator:
    def __init__(self):
        self.wx = 0.0
        self.wy = 0.0
        self.alpha = WIND_ALPHA
        self.error_samples = []
        self.vector_error = []

    def update(self, ground_vx, ground_vy, heading_deg,
               airspeed, health=1.0):
        h = math.radians(heading_deg)

        # Air-relative velocity in navigation frame.
        avx = airspeed * math.cos(h)
        avy = airspeed * math.sin(h)

        raw_wx = ground_vx - avx
        raw_wy = ground_vy - avy

        raw_wx = clamp(raw_wx, -MAX_WIND_ESTIMATE, MAX_WIND_ESTIMATE)
        raw_wy = clamp(raw_wy, -MAX_WIND_ESTIMATE, MAX_WIND_ESTIMATE)

        adaptive_alpha = self.alpha * (0.35 + 0.65 * health)

        self.wx += adaptive_alpha * (raw_wx - self.wx)
        self.wy += adaptive_alpha * (raw_wy - self.wy)

        residual = math.hypot(raw_wx - self.wx, raw_wy - self.wy)
        self.error_samples.append(residual)

        recent = self.error_samples[-30:]
        if recent:
            uncertainty = statistics.mean(recent)
        else:
            uncertainty = WIND_UNCERTAINTY_FLOOR

        uncertainty = max(WIND_UNCERTAINTY_FLOOR,
                          min(4.0, uncertainty))
        return self.wx, self.wy, uncertainty

    @property
    def speed(self):
        return math.hypot(self.wx, self.wy)

    @property
    def direction(self):
        return angle_deg(self.wx, self.wy)


# =========================
# SIMPLE EKF-LIKE NAVIGATION
# =========================
class NavigationFilter:
    """
    Lightweight EKF-style navigation filter for the virtual study.
    State:
        x, y, altitude, heading_deg, vx, vy
    """
    def __init__(self):
        self.x = INITIAL_X
        self.y = INITIAL_Y
        self.altitude = INITIAL_ALTITUDE
        self.heading = 0.0
        self.vx = HORIZONTAL_AIR_SPEED
        self.vy = 0.0

        self.pos_var = 9.0
        self.alt_var = 4.0
        self.heading_var = 4.0

        self.pos_errors = []
        self.alt_errors = []
        self.heading_errors = []

    def predict(self, dt, steering, wind_x, wind_y):
        turn_rate = steering * MAX_TURN_RATE_DEG_S
        self.heading = (self.heading + turn_rate * dt) % 360.0

        h = math.radians(self.heading)
        self.vx = HORIZONTAL_AIR_SPEED * math.cos(h) + wind_x
        self.vy = HORIZONTAL_AIR_SPEED * math.sin(h) + wind_y

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.altitude = max(0.0, self.altitude - VERTICAL_DESCENT * dt)

        self.pos_var += 0.02 * dt
        self.alt_var += 0.015 * dt
        self.heading_var += 0.01 * dt

    def gnss_update(self, mx, my, mvx, mvy, health):
        weight = clamp(health, 0.05, 1.0)

        k = (self.pos_var * weight) / (self.pos_var * weight + GNSS_POS_NOISE ** 2)
        self.x += k * (mx - self.x)
        self.y += k * (my - self.y)
        self.pos_var *= (1.0 - k)

        kv = 0.35 * weight
        self.vx += kv * (mvx - self.vx)
        self.vy += kv * (mvy - self.vy)

    def baro_update(self, mz, health):
        r = BARO_NOISE ** 2 / max(health, 0.05)
        k = self.alt_var / (self.alt_var + r)
        self.altitude += k * (mz - self.altitude)
        self.alt_var *= (1.0 - k)

    def imu_update(self, heading_measurement, health):
        r = IMU_HEADING_NOISE_DEG ** 2 / max(health, 0.05)
        k = self.heading_var / (self.heading_var + r)
        error = wrap_angle_deg(heading_measurement - self.heading)
        self.heading = (self.heading + k * error) % 360.0
        self.heading_var *= (1.0 - k)

    def record_error(self, true_state):
        self.pos_errors.append(
            distance(self.x, self.y, true_state.x, true_state.y)
        )
        self.alt_errors.append(abs(self.altitude - true_state.altitude))
        self.heading_errors.append(
            abs(wrap_angle_deg(self.heading - true_state.heading))
        )


# =========================
# TRUE VEHICLE
# =========================
@dataclass
class Vehicle:
    x: float = INITIAL_X
    y: float = INITIAL_Y
    altitude: float = INITIAL_ALTITUDE
    heading: float = 0.0
    steering: float = 0.0


# =========================
# REACHABILITY / TRAJECTORY
# =========================
def simulate_candidate(x0, y0, altitude, heading_deg,
                       steering_cmd, wind_x, wind_y,
                       wind_uncertainty, horizon):
    """
    Propagate a candidate steering command to touchdown.

    Steering is normalized:
        -1 = left
        +1 = right

    The candidate heading is rate-limited. Wind is included explicitly.
    """
    x = x0
    y = y0
    h = heading_deg

    t = 0.0
    dt = CANDIDATE_SIM_DT

    while t < min(horizon, MAX_PREDICTION_TIME) and altitude > 0.0:
        rate = steering_cmd * MAX_TURN_RATE_DEG_S
        h = (h + rate * dt) % 360.0

        hr = math.radians(h)
        vx = HORIZONTAL_AIR_SPEED * math.cos(hr) + wind_x
        vy = HORIZONTAL_AIR_SPEED * math.sin(hr) + wind_y

        x += vx * dt
        y += vy * dt
        altitude -= VERTICAL_DESCENT * dt
        t += dt

    if altitude > 0:
        remaining = altitude / VERTICAL_DESCENT
        hr = math.radians(h)
        x += (HORIZONTAL_AIR_SPEED * math.cos(hr) + wind_x) * remaining
        y += (HORIZONTAL_AIR_SPEED * math.sin(hr) + wind_y) * remaining
        t += remaining
        altitude = 0.0

    return x, y, h, t


def cross_track_error(x, y, heading_deg):
    """
    Signed cross-track error relative to target line.
    """
    dx = TARGET_X - x
    dy = TARGET_Y - y

    desired = angle_deg(dx, dy)
    return wrap_angle_deg(desired - heading_deg)


def candidate_cost(x, y, h, steering_cmd,
                   current_steering, wind_uncertainty,
                   altitude):
    d = distance(x, y, TARGET_X, TARGET_Y)

    # Cross-track component
    cte = abs(cross_track_error(x, y, h))

    # Overshoot penalty. Stronger at lower altitude.
    overshoot = 0.0
    target_dx = TARGET_X - INITIAL_X
    target_dy = TARGET_Y - INITIAL_Y
    path_len = math.hypot(target_dx, target_dy)

    travelled_projection = (
        (x - INITIAL_X) * target_dx +
        (y - INITIAL_Y) * target_dy
    ) / max(path_len, 1.0)

    if travelled_projection > path_len:
        overshoot = travelled_projection - path_len

    # Steering smoothness
    steering_change = abs(steering_cmd - current_steering)

    # Heading error at predicted touchdown
    desired_final = angle_deg(TARGET_X - x, TARGET_Y - y)
    heading_error = abs(wrap_angle_deg(desired_final - h))

    # Wind uncertainty creates a penalty for candidates near the target
    # if uncertainty could move touchdown substantially.
    uncertainty_penalty = wind_uncertainty * (
        0.15 + 0.003 * d
    )

    # Near touchdown, position dominates.
    altitude_factor = clamp(1.0 - altitude / INITIAL_ALTITUDE, 0.0, 1.0)
    position_weight = W_POSITION * (1.0 + 2.0 * altitude_factor)

    return (
        position_weight * d
        + W_CROSS_TRACK * cte
        + W_HEADING * (heading_error / 45.0)
        + W_STEERING * steering_change
        + W_OVERSHOOT * overshoot
        + W_WIND_UNCERTAINTY * uncertainty_penalty
    )


def choose_reachable_command(nav, wind_estimator, current_steering):
    horizon = adaptive_horizon(nav.altitude)

    commands = np.linspace(
        -MAX_STEERING,
        MAX_STEERING,
        NUM_COMMANDS
    )

    best = None

    for cmd in commands:
        px, py, ph, pt = simulate_candidate(
            nav.x,
            nav.y,
            nav.altitude,
            nav.heading,
            float(cmd),
            wind_estimator.wx,
            wind_estimator.wy,
            max(wind_estimator.error_samples[-1:]
                or [WIND_UNCERTAINTY_FLOOR]),
            horizon
        )

        cost = candidate_cost(
            px, py, ph, float(cmd),
            current_steering,
            max(wind_estimator.error_samples[-1:]
                or [WIND_UNCERTAINTY_FLOOR]),
            nav.altitude
        )

        result = {
            "command": float(cmd),
            "x": px,
            "y": py,
            "heading": ph,
            "time": pt,
            "cost": cost
        }

        if best is None or cost < best["cost"]:
            best = result

    return best, horizon


def limit_steering(current, target, dt):
    max_change = STEERING_RATE_PER_SEC * dt

    if target > current:
        return min(target, current + max_change)
    return max(target, current - max_change)


# =========================
# SIMULATION
# =========================
def run_simulation(wind_speed,
                   wind_dir,
                   seed=109,
                   print_progress=False):
    rng = random.Random(seed)

    wx_true, wy_true = wind_components(wind_speed, wind_dir)

    true = Vehicle()
    nav = NavigationFilter()
    health = SensorHealth()
    estimator = WindEstimator()

    t = 0.0
    dt = 1.0 / IMU_RATE

    next_gnss = 0.0
    next_baro = 0.0
    next_airspeed = 0.0
    next_guidance = 0.0

    command = 0.0
    guidance_commands = []
    wind_errors = []

    steering_reversals = 0
    previous_command_sign = 0

    max_position_error = 0.0

    while true.altitude > 0.0 and t < 300.0:

        # ---------------------------------
        # TRUE VEHICLE
        # ---------------------------------
        turn_rate = command * MAX_TURN_RATE_DEG_S
        true.heading = (true.heading + turn_rate * dt) % 360.0

        hr = math.radians(true.heading)

        air_vx = HORIZONTAL_AIR_SPEED * math.cos(hr)
        air_vy = HORIZONTAL_AIR_SPEED * math.sin(hr)

        true_vx = air_vx + wx_true
        true_vy = air_vy + wy_true

        true.x += true_vx * dt
        true.y += true_vy * dt
        true.altitude = max(
            0.0,
            true.altitude - VERTICAL_DESCENT * dt
        )

        # ---------------------------------
        # SENSOR UPDATES
        # ---------------------------------
        if t >= next_gnss:
            mx = true.x + rng.gauss(0, GNSS_POS_NOISE)
            my = true.y + rng.gauss(0, GNSS_POS_NOISE)

            mvx = true_vx + rng.gauss(0, GNSS_VEL_NOISE)
            mvy = true_vy + rng.gauss(0, GNSS_VEL_NOISE)

            nav.gnss_update(
                mx, my, mvx, mvy,
                health.gnss
            )

            next_gnss += 1.0 / GNSS_RATE

        if t >= next_baro:
            mz = true.altitude + rng.gauss(0, BARO_NOISE)

            nav.baro_update(
                mz,
                health.baro
            )

            next_baro += 1.0 / BARO_RATE

        if t >= next_airspeed:
            measured_airspeed = (
                AIR_SPEED +
                rng.gauss(0, AIRSPEED_NOISE)
            )

            # Wind estimation uses GNSS ground velocity and
            # IMU heading. The airspeed measurement is used here.
            if health.gnss > 0.2 and health.airspeed > 0.2:
                estimator.update(
                    nav.vx,
                    nav.vy,
                    nav.heading,
                    measured_airspeed,
                    health.airspeed
                )

            next_airspeed += 1.0 / AIRSPEED_RATE

        # IMU update
        measured_heading = (
            true.heading +
            rng.gauss(0, IMU_HEADING_NOISE_DEG)
        )
        nav.imu_update(
            measured_heading,
            health.imu
        )

        # ---------------------------------
        # NAVIGATION PROPAGATION
        # ---------------------------------
        nav.predict(
            dt,
            command,
            estimator.wx,
            estimator.wy
        )

        nav.record_error(true)

        current_pos_error = distance(
            nav.x, nav.y,
            true.x, true.y
        )
        max_position_error = max(
            max_position_error,
            current_pos_error
        )

        # ---------------------------------
        # WIND ERROR
        # ---------------------------------
        wind_err = math.hypot(
            estimator.wx - wx_true,
            estimator.wy - wy_true
        )
        wind_errors.append(wind_err)

        # ---------------------------------
        # GUIDANCE
        # ---------------------------------
        if t >= next_guidance and true.altitude > 0:

            best, horizon = choose_reachable_command(
                nav,
                estimator,
                command
            )

            desired_command = best["command"]

            # Target capture logic:
            # if PTP is already inside tolerance, reduce command
            # and avoid unnecessary oscillations.
            ptp_error = distance(
                best["x"],
                best["y"],
                TARGET_X,
                TARGET_Y
            )

            if ptp_error <= TARGET_CAPTURE_RADIUS:
                desired_command *= 0.35

            new_command = limit_steering(
                command,
                desired_command,
                GUIDANCE_INTERVAL
            )

            old_sign = 0 if abs(command) < 0.02 else (
                1 if command > 0 else -1
            )
            new_sign = 0 if abs(new_command) < 0.02 else (
                1 if new_command > 0 else -1
            )

            if old_sign != 0 and new_sign != 0 and old_sign != new_sign:
                steering_reversals += 1

            command = clamp(
                new_command,
                -MAX_STEERING,
                MAX_STEERING
            )

            guidance_commands.append(command)
            next_guidance += GUIDANCE_INTERVAL

        t += dt

    # ---------------------------------
    # FINAL ESTIMATES
    # ---------------------------------
    true_error = distance(
        true.x, true.y,
        TARGET_X, TARGET_Y
    )

    estimated_error = distance(
        nav.x, nav.y,
        TARGET_X, TARGET_Y
    )

    rms_pos = math.sqrt(
        statistics.mean(e * e for e in nav.pos_errors)
    )
    rms_alt = math.sqrt(
        statistics.mean(e * e for e in nav.alt_errors)
    )
    rms_heading = math.sqrt(
        statistics.mean(e * e for e in nav.heading_errors)
    )

    mean_wind_error = statistics.mean(wind_errors)
    rms_wind_error = math.sqrt(
        statistics.mean(e * e for e in wind_errors)
    )

    final_wind_error = math.hypot(
        estimator.wx - wx_true,
        estimator.wy - wy_true
    )

    return {
        "true_x": true.x,
        "true_y": true.y,
        "estimated_x": nav.x,
        "estimated_y": nav.y,
        "true_landing_error": true_error,
        "estimated_landing_error": estimated_error,
        "flight_time": t,

        "wind_x_true": wx_true,
        "wind_y_true": wy_true,
        "wind_speed_true": wind_speed,
        "wind_direction_true": wind_dir,

        "wind_x_est": estimator.wx,
        "wind_y_est": estimator.wy,
        "wind_speed_est": estimator.speed,
        "wind_direction_est": estimator.direction,

        "mean_wind_error": mean_wind_error,
        "rms_wind_error": rms_wind_error,
        "final_wind_error": final_wind_error,

        "ekf_rms_position": rms_pos,
        "ekf_rms_altitude": rms_alt,
        "ekf_rms_heading": rms_heading,
        "max_position_error": max_position_error,

        "final_gnss_health": health.gnss,
        "final_imu_health": health.imu,
        "final_baro_health": health.baro,
        "final_airspeed_health": health.airspeed,

        "average_steering": (
            statistics.mean(abs(c) for c in guidance_commands)
            if guidance_commands else 0.0
        ),
        "steering_reversals": steering_reversals,

        "status": true_error <= TOLERANCE,
    }


# =========================
# PRINTING
# =========================
def print_header():
    print("=" * 72)
    print("VIRTUAL PARAFOIL V10.9")
    print("REACHABILITY-BASED WIND-AWARE GUIDANCE")
    print("=" * 72)
    print()
    print("PARAFOIL PARAMETERS")
    print(f"Area: {AREA:.2f} m^2")
    print(f"Mass: {MASS:.1f} kg")
    print(f"CL: {CL:.2f}")
    print(f"CD: {CD:.2f}")
    print(f"Airspeed: {AIR_SPEED:.6f} m/s")
    print(f"Horizontal air velocity: {HORIZONTAL_AIR_SPEED:.6f} m/s")
    print(f"Vertical descent velocity: {VERTICAL_DESCENT:.3f} m/s")
    print(f"Glide ratio: {GLIDE_RATIO:.3f}")
    print()
    print("TARGET")
    print(f"Target X: {TARGET_X:.1f} m")
    print(f"Target Y: {TARGET_Y:.1f} m")
    print(f"Tolerance: {TOLERANCE:.1f} m")
    print()
    print("REFERENCE WIND")
    print(f"Wind speed: {REF_WIND_SPEED:.1f} m/s")
    print(f"Wind direction: {REF_WIND_DIR_DEG:.1f} degrees")
    print()
    print("SENSOR MODEL")
    print(f"GNSS position noise: {GNSS_POS_NOISE:.1f} m")
    print(f"GNSS velocity noise: {GNSS_VEL_NOISE:.2f} m/s")
    print(f"Barometer noise: {BARO_NOISE:.1f} m")
    print(f"IMU heading noise: {IMU_HEADING_NOISE_DEG:.1f} deg")
    print(f"IMU turn-rate noise: {IMU_TURN_NOISE_DEG_S:.1f} deg/s")
    print(f"Airspeed noise: {AIRSPEED_NOISE:.2f} m/s")
    print()
    print("UPDATE RATES")
    print(f"GNSS: {GNSS_RATE:.1f} Hz")
    print(f"Barometer: {BARO_RATE:.1f} Hz")
    print(f"IMU: {IMU_RATE:.1f} Hz")
    print(f"Airspeed: {AIRSPEED_RATE:.1f} Hz")
    print()
    print("V10.9 DEVELOPMENT")
    print("Online wind estimation: ENABLED")
    print("Wind uncertainty estimation: ENABLED")
    print("Predicted touchdown point: ENABLED")
    print("Reachability-based guidance: ENABLED")
    print("Wind-compensated guidance: ENABLED")
    print("Cross-track guidance: ENABLED")
    print("Candidate trajectory search: ENABLED")
    print("Overshoot protection: ENABLED")
    print("Target capture mode: ENABLED")
    print("Adaptive prediction horizon: ENABLED")
    print("Steering rate limiting: ENABLED")
    print()


def print_reference(r):
    print("=" * 72)
    print("STARTING V10.9 REFERENCE WIND SIMULATION")
    print("=" * 72)
    print()
    print("=" * 72)
    print("V10.9 REFERENCE RESULTS")
    print("=" * 72)
    print(f"True landing X: {r['true_x']:.3f} m")
    print(f"True landing Y: {r['true_y']:.3f} m")
    print(f"Estimated landing X: {r['estimated_x']:.3f} m")
    print(f"Estimated landing Y: {r['estimated_y']:.3f} m")
    print(f"Target: {TARGET_X:.3f}, {TARGET_Y:.3f} m")
    print(f"True landing error: {r['true_landing_error']:.3f} m")
    print(f"Estimated landing error: {r['estimated_landing_error']:.3f} m")
    print(f"Flight time: {r['flight_time']:.3f} s")
    print()
    print("--- WIND ESTIMATION ---")
    print(f"True wind X: {r['wind_x_true']:.3f} m/s")
    print(f"True wind Y: {r['wind_y_true']:.3f} m/s")
    print(f"True wind speed: {r['wind_speed_true']:.3f} m/s")
    print(f"True wind direction: {r['wind_direction_true']:.3f} deg")
    print(f"Estimated wind X: {r['wind_x_est']:.3f} m/s")
    print(f"Estimated wind Y: {r['wind_y_est']:.3f} m/s")
    print(f"Estimated wind speed: {r['wind_speed_est']:.3f} m/s")
    print(f"Estimated wind direction: {r['wind_direction_est']:.3f} deg")
    print(f"Mean wind estimation error: {r['mean_wind_error']:.3f} m/s")
    print(f"RMS wind estimation error: {r['rms_wind_error']:.3f} m/s")
    print(f"Final wind-vector error: {r['final_wind_error']:.3f} m/s")
    print()
    print("--- NAVIGATION ---")
    print(f"EKF RMS position error: {r['ekf_rms_position']:.3f} m")
    print(f"EKF RMS altitude error: {r['ekf_rms_altitude']:.3f} m")
    print(f"EKF RMS heading error: {r['ekf_rms_heading']:.3f} deg")
    print()
    print("--- GUIDANCE ---")
    print(f"Average steering: {r['average_steering']:.3f}")
    print(f"Steering reversals: {r['steering_reversals']}")
    print(f"Landing status: {fmt_status(r['true_landing_error'])}")
    print()


def run_direction_study():
    print("=" * 72)
    print("V10.9 WIND DIRECTION STUDY")
    print("=" * 72)
    print(f"Wind speed fixed at {REF_WIND_SPEED:.1f} m/s")

    results = []

    for i, direction in enumerate(DIRECTION_STUDY, 1):
        r = run_simulation(
            REF_WIND_SPEED,
            float(direction),
            seed=1000 + i
        )
        results.append((direction, r))

        print(
            f"Simulation {i}/8 | Direction = {direction:6.1f} deg | "
            f"Landing Error = {r['true_landing_error']:8.3f} m | "
            f"Wind RMS = {r['rms_wind_error']:7.3f} m/s"
        )

    return results


def run_speed_study():
    print()
    print("=" * 72)
    print("V10.9 WIND SPEED STUDY")
    print("=" * 72)
    print("Wind direction fixed at 0.0 degrees")

    results = []

    for i, speed in enumerate(SPEED_STUDY, 1):
        r = run_simulation(
            float(speed),
            0.0,
            seed=2000 + i
        )
        results.append((speed, r))

        print(
            f"Simulation {i}/8 | Wind = {speed:5.2f} m/s | "
            f"Landing Error = {r['true_landing_error']:8.3f} m | "
            f"Estimated Wind = {r['wind_speed_est']:6.3f} m/s | "
            f"Wind RMS = {r['rms_wind_error']:7.3f} m/s"
        )

    return results


def run_monte_carlo():
    print()
    print("=" * 72)
    print("V10.9 MONTE CARLO WIND VALIDATION")
    print("=" * 72)
    print(f"Number of simulations: {MC_RUNS}")
    print("Wind speed range: 0.0 - 7.0 m/s")
    print("Wind direction range: 0 - 360 degrees")

    rng = random.Random(MC_SEED)
    records = []

    for i in range(1, MC_RUNS + 1):
        speed = rng.uniform(0.0, 7.0)
        direction = rng.uniform(0.0, 360.0)

        r = run_simulation(
            speed,
            direction,
            seed=MC_SEED + i * 17
        )

        records.append((speed, direction, r))

        if i == 1 or i % 10 == 0 or i == MC_RUNS:
            print(
                f"Simulation {i:3d}/{MC_RUNS} | "
                f"Wind = {speed:5.2f} m/s | "
                f"Direction = {direction:6.1f} deg | "
                f"Landing Error = {r['true_landing_error']:8.2f} m | "
                f"Wind RMS = {r['rms_wind_error']:5.2f} m/s"
            )

    return records


def print_mc_results(records):
    landing_errors = [x[2]["true_landing_error"] for x in records]
    wind_errors = [x[2]["rms_wind_error"] for x in records]

    print()
    print("=" * 72)
    print("V10.9 MONTE CARLO RESULTS")
    print("=" * 72)

    mean_landing = statistics.mean(landing_errors)
    median_landing = statistics.median(landing_errors)
    std_landing = statistics.pstdev(landing_errors)

    print(f"Number of simulations: {len(records)}")
    print(f"Mean landing error: {mean_landing:.3f} m")
    print(f"Median landing error: {median_landing:.3f} m")
    print(f"Standard deviation: {std_landing:.3f} m")
    print(f"Minimum landing error: {min(landing_errors):.3f} m")
    print(f"Maximum landing error: {max(landing_errors):.3f} m")
    print(f"Landing within 5 m: "
          f"{100.0 * sum(e <= 5 for e in landing_errors) / len(records):.2f} %")
    print(f"Landing within 10 m: "
          f"{100.0 * sum(e <= 10 for e in landing_errors) / len(records):.2f} %")
    print(f"Landing within 20 m: "
          f"{100.0 * sum(e <= 20 for e in landing_errors) / len(records):.2f} %")
    print(f"Landing within 50 m: "
          f"{100.0 * sum(e <= 50 for e in landing_errors) / len(records):.2f} %")
    print(f"Landing within 100 m: "
          f"{100.0 * sum(e <= 100 for e in landing_errors) / len(records):.2f} %")
    print(f"Mean wind RMS error: {statistics.mean(wind_errors):.3f} m/s")
    print(f"Median wind RMS error: {statistics.median(wind_errors):.3f} m/s")

    best_landing = min(records, key=lambda x: x[2]["true_landing_error"])
    worst_landing = max(records, key=lambda x: x[2]["true_landing_error"])
    best_wind = min(records, key=lambda x: x[2]["rms_wind_error"])
    worst_wind = max(records, key=lambda x: x[2]["rms_wind_error"])

    for title, rec in [
        ("BEST LANDING CASE", best_landing),
        ("WORST LANDING CASE", worst_landing),
    ]:
        speed, direction, r = rec
        print()
        print(title)
        print(f"Wind speed: {speed:.3f} m/s")
        print(f"Wind direction: {direction:.3f} degrees")
        print(f"Wind X: {r['wind_x_true']:.3f} m/s")
        print(f"Wind Y: {r['wind_y_true']:.3f} m/s")
        print(f"Landing X: {r['true_x']:.3f} m")
        print(f"Landing Y: {r['true_y']:.3f} m")
        print(f"Landing error: {r['true_landing_error']:.3f} m")

    print()
    print("BEST WIND ESTIMATION CASE")
    print(f"Wind speed: {best_wind[0]:.3f} m/s")
    print(f"Wind direction: {best_wind[1]:.3f} degrees")
    print(f"Wind RMS estimation error: "
          f"{best_wind[2]['rms_wind_error']:.3f} m/s")

    print()
    print("WORST WIND ESTIMATION CASE")
    print(f"Wind speed: {worst_wind[0]:.3f} m/s")
    print(f"Wind direction: {worst_wind[1]:.3f} degrees")
    print(f"Wind RMS estimation error: "
          f"{worst_wind[2]['rms_wind_error']:.3f} m/s")

    return {
        "mean_landing": mean_landing,
        "median_landing": median_landing,
        "std_landing": std_landing,
        "min_landing": min(landing_errors),
        "max_landing": max(landing_errors),
        "success20": 100.0 * sum(e <= 20 for e in landing_errors) / len(records),
        "mean_wind": statistics.mean(wind_errors),
        "median_wind": statistics.median(wind_errors),
        "best_landing": best_landing,
        "worst_landing": worst_landing,
    }


def print_assessment(reference, mc):
    print()
    print("=" * 72)
    print("V10.9 NAVIGATION + GUIDANCE SYSTEM ASSESSMENT")
    print("=" * 72)
    print(f"Raw GNSS position noise: {GNSS_POS_NOISE:.1f} m")
    print(f"Reference wind RMS error: "
          f"{reference['rms_wind_error']:.3f} m/s")
    print(f"Reference landing error: "
          f"{reference['true_landing_error']:.3f} m")
    print(f"Monte Carlo mean wind RMS error: "
          f"{mc['mean_wind']:.3f} m/s")
    print(f"Monte Carlo landing success <=20 m: "
          f"{mc['success20']:.2f} %")
    print("ONLINE WIND ESTIMATION: ACTIVE")
    print("WIND UNCERTAINTY MODEL: ACTIVE")
    print("PREDICTED TOUCHDOWN POINT: ACTIVE")
    print("REACHABILITY-BASED GUIDANCE: ACTIVE")
    print("WIND-COMPENSATED GUIDANCE: ACTIVE")
    print("CROSS-TRACK GUIDANCE: ACTIVE")
    print("CANDIDATE TRAJECTORY SEARCH: ACTIVE")
    print("OVERSHOOT PROTECTION: ACTIVE")
    print("TARGET CAPTURE MODE: ACTIVE")
    print("ADAPTIVE PREDICTION HORIZON: ACTIVE")
    print("STEERING RATE LIMITING: ACTIVE")
    print()
    print("V10.9 STATUS: COMPLETE")


def main():
    print_header()

    reference = run_simulation(
        REF_WIND_SPEED,
        REF_WIND_DIR_DEG,
        seed=109
    )

    print_reference(reference)

    run_direction_study()
    run_speed_study()

    mc_records = run_monte_carlo()
    mc = print_mc_results(mc_records)

    print_assessment(reference, mc)

    print()
    print("=" * 72)
    print("V10.9 SIMULATION COMPLETE")
    print("=" * 72)
    print(f"Reference wind: {REF_WIND_SPEED:.1f} m/s")
    print(f"Reference direction: {REF_WIND_DIR_DEG:.1f} degrees")
    print("Wind estimator: ONLINE")
    print("Reachability-based guidance: ACTIVE")
    print("Predicted touchdown point: ACTIVE")
    print("Wind-aware guidance: ACTIVE")
    print("Monte Carlo simulations: 100")
    print("=" * 72)


if __name__ == "__main__":
    main()
    print("BHAYANKAR CHANGE")
    print("Hi change")