"""
============================================================
VIRTUAL PARAFOIL V10.3
GNSS FAILURE-DURATION ROBUSTNESS STUDY
============================================================

Development progression:
V8.x  -> Guidance development
V9.x  -> Reachability-aware guidance
V10.0 -> Sensor noise + EKF
V10.1 -> Sensor update-rate study
V10.2 -> GNSS dropout + outlier rejection
V10.3 -> Continuous GNSS failure-duration robustness

Purpose:
Evaluate how the parafoil navigation/guidance system behaves
when GNSS is continuously unavailable for different durations.

GNSS outage durations tested:
    0 s
    2 s
    5 s
    10 s
    20 s
    30 s
    60 s

Sensors:
    GNSS
    Barometer
    IMU heading / turn rate

Estimator:
    Extended Kalman Filter

Guidance:
    Short-horizon candidate steering search
    Adaptive prediction horizon
    Wind-aware prediction

NOTE:
This is a simulation model for GNC development.
It is NOT flight-certified hardware/software.
============================================================
"""

import math
import numpy as np


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

AREA = 0.96
MASS = 1.0
CL = 0.4
CD = 0.25

AIR_DENSITY = 1.225
GRAVITY = 9.81

INITIAL_ALTITUDE = 600.0

TARGET_X = 500.0
TARGET_Y = 200.0

TOLERANCE = 20.0

# Parafoil horizontal airspeed
AIRSPEED = 7.0132343805211725

# Derived horizontal velocity from previous versions
HORIZONTAL_AIR_VELOCITY = 5.947210860272128

# Vertical descent velocity
VERTICAL_DESCENT_VELOCITY = 3.71700678767008

GLIDE_RATIO = (
    HORIZONTAL_AIR_VELOCITY /
    VERTICAL_DESCENT_VELOCITY
)


# ============================================================
# WIND
# ============================================================

WIND_SPEED = 3.0
WIND_DIRECTION_DEG = 0.0

WIND_DIRECTION_RAD = math.radians(WIND_DIRECTION_DEG)

WIND_X = WIND_SPEED * math.cos(WIND_DIRECTION_RAD)
WIND_Y = WIND_SPEED * math.sin(WIND_DIRECTION_RAD)


# ============================================================
# SIMULATION TIMING
# ============================================================

DT = 0.02

MAX_TIME = 250.0

GUIDANCE_INTERVAL = 2.0

IMU_RATE = 50.0
GNSS_RATE = 5.0
BARO_RATE = 10.0

IMU_DT = 1.0 / IMU_RATE
GNSS_DT = 1.0 / GNSS_RATE
BARO_DT = 1.0 / BARO_RATE


# ============================================================
# GUIDANCE PARAMETERS
# ============================================================

NUM_CANDIDATES = 21

STEERING_COMMANDS = np.linspace(
    -1.0,
    1.0,
    NUM_CANDIDATES
)

MAX_TURN_RATE_DEG = 15.0
MAX_TURN_RATE = math.radians(MAX_TURN_RATE_DEG)

POSITION_WEIGHT = 0.65
HEADING_WEIGHT = 0.35


# ============================================================
# ADAPTIVE HORIZON
# ============================================================

def get_prediction_horizon(altitude):
    """
    Adaptive prediction horizon based on altitude.
    """

    if altitude > 400.0:
        return 20.0

    elif altitude > 200.0:
        return 15.0

    elif altitude > 100.0:
        return 10.0

    else:
        return 5.0


# ============================================================
# SENSOR MODEL
# ============================================================

GNSS_POSITION_NOISE = 3.0
GNSS_VELOCITY_NOISE = 0.3

BARO_ALTITUDE_NOISE = 2.0

IMU_HEADING_NOISE_DEG = 2.0
IMU_TURN_RATE_NOISE_DEG = 0.5

IMU_HEADING_NOISE = math.radians(
    IMU_HEADING_NOISE_DEG
)

IMU_TURN_RATE_NOISE = math.radians(
    IMU_TURN_RATE_NOISE_DEG
)


# ============================================================
# GNSS OUTLIER PARAMETERS
# ============================================================

OUTLIER_REJECTION_ENABLED = True

GNSS_OUTLIER_PROBABILITY = 0.05

POSITION_OUTLIER_MAGNITUDE = 40.0
VELOCITY_OUTLIER_MAGNITUDE = 5.0


# ============================================================
# EKF INITIALIZATION
# ============================================================

# State:
#
# x = [
#       x,
#       y,
#       vx,
#       vy,
#       altitude,
#       heading,
#       turn_rate
#     ]

STATE_SIZE = 7


def wrap_angle(angle):
    """
    Wrap angle into [-pi, pi].
    """

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class EKF:
    """
    Simplified Extended Kalman Filter for parafoil navigation.
    """

    def __init__(self):

        self.x = np.array([
            0.0,
            0.0,
            HORIZONTAL_AIR_VELOCITY,
            0.0,
            INITIAL_ALTITUDE,
            0.0,
            0.0
        ], dtype=float)

        self.P = np.diag([
            25.0,
            25.0,
            4.0,
            4.0,
            16.0,
            math.radians(10.0) ** 2,
            math.radians(5.0) ** 2
        ])

        self.Q = np.diag([
            0.05,
            0.05,
            0.20,
            0.20,
            0.10,
            math.radians(0.5) ** 2,
            math.radians(0.5) ** 2
        ])

        self.R_gnss = np.diag([
            GNSS_POSITION_NOISE ** 2,
            GNSS_POSITION_NOISE ** 2,
            GNSS_VELOCITY_NOISE ** 2,
            GNSS_VELOCITY_NOISE ** 2
        ])

        self.R_baro = np.array([
            [BARO_ALTITUDE_NOISE ** 2]
        ])

        self.R_imu = np.diag([
            IMU_HEADING_NOISE ** 2,
            IMU_TURN_RATE_NOISE ** 2
        ])

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(self, dt):

        x = self.x.copy()

        px = x[0]
        py = x[1]

        vx = x[2]
        vy = x[3]

        altitude = x[4]

        heading = x[5]
        turn_rate = x[6]

        new_heading = wrap_angle(
            heading + turn_rate * dt
        )

        self.x[0] = px + vx * dt
        self.x[1] = py + vy * dt

        self.x[2] = vx
        self.x[3] = vy

        self.x[4] = max(
            0.0,
            altitude -
            VERTICAL_DESCENT_VELOCITY * dt
        )

        self.x[5] = new_heading
        self.x[6] = turn_rate

        # Numerical Jacobian / simplified state transition
        F = np.eye(STATE_SIZE)

        F[0, 2] = dt
        F[1, 3] = dt
        F[5, 6] = dt

        self.P = (
            F @ self.P @ F.T
            + self.Q * dt
        )

    # --------------------------------------------------------
    # GNSS UPDATE
    # --------------------------------------------------------

    def update_gnss(
        self,
        measurement,
        stats
    ):
        """
        measurement:
        [x, y, vx, vy]
        """

        H = np.zeros(
            (4, STATE_SIZE)
        )

        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0

        z = np.array(
            measurement,
            dtype=float
        )

        z_pred = H @ self.x

        innovation = z - z_pred

        S = (
            H @ self.P @ H.T
            + self.R_gnss
        )

        # Mahalanobis distance
        try:
            mahalanobis = float(
                innovation.T
                @ np.linalg.inv(S)
                @ innovation
            )
        except np.linalg.LinAlgError:
            mahalanobis = 999.0

        # 4 DOF chi-square style threshold
        threshold = 13.28

        if (
            OUTLIER_REJECTION_ENABLED
            and mahalanobis > threshold
        ):

            stats["gnss_rejected"] += 1

            return False

        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )

        self.x = (
            self.x
            + K @ innovation
        )

        I = np.eye(STATE_SIZE)

        self.P = (
            (I - K @ H)
            @ self.P
        )

        self.x[5] = wrap_angle(
            self.x[5]
        )

        stats["gnss_accepted"] += 1

        return True

    # --------------------------------------------------------
    # BAROMETER UPDATE
    # --------------------------------------------------------

    def update_baro(
        self,
        altitude_measurement
    ):

        H = np.zeros(
            (1, STATE_SIZE)
        )

        H[0, 4] = 1.0

        z = np.array([
            altitude_measurement
        ])

        z_pred = H @ self.x

        innovation = z - z_pred

        S = (
            H @ self.P @ H.T
            + self.R_baro
        )

        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )

        self.x = (
            self.x
            + (
                K @ innovation
            ).flatten()
        )

        I = np.eye(STATE_SIZE)

        self.P = (
            (I - K @ H)
            @ self.P
        )

        self.x[4] = max(
            0.0,
            self.x[4]
        )

    # --------------------------------------------------------
    # IMU UPDATE
    # --------------------------------------------------------

    def update_imu(
        self,
        heading_measurement,
        turn_rate_measurement
    ):

        H = np.zeros(
            (2, STATE_SIZE)
        )

        H[0, 5] = 1.0
        H[1, 6] = 1.0

        z = np.array([
            heading_measurement,
            turn_rate_measurement
        ])

        z_pred = H @ self.x

        innovation = z - z_pred

        innovation[0] = wrap_angle(
            innovation[0]
        )

        S = (
            H @ self.P @ H.T
            + self.R_imu
        )

        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )

        self.x = (
            self.x
            + K @ innovation
        )

        I = np.eye(STATE_SIZE)

        self.P = (
            (I - K @ H)
            @ self.P
        )

        self.x[5] = wrap_angle(
            self.x[5]
        )


# ============================================================
# PARAFOIL STATE
# ============================================================

class ParafoilState:

    def __init__(self):

        self.x = 0.0
        self.y = 0.0

        self.altitude = INITIAL_ALTITUDE

        self.heading = 0.0

        self.turn_rate = 0.0

        self.steering = 0.0

        self.vx = (
            HORIZONTAL_AIR_VELOCITY
            + WIND_X
        )

        self.vy = WIND_Y


# ============================================================
# PARAFOIL DYNAMICS
# ============================================================

def propagate_true_state(
    state,
    steering,
    dt,
    wind_x,
    wind_y
):
    """
    Propagate the true parafoil state.

    steering:
        -1 = full left
         0 = neutral
        +1 = full right
    """

    # Desired turn rate
    desired_turn_rate = (
        steering
        * MAX_TURN_RATE
    )

    # First-order turn-rate response
    turn_response = 4.0

    state.turn_rate += (
        desired_turn_rate
        - state.turn_rate
    ) * turn_response * dt

    state.heading = wrap_angle(
        state.heading
        + state.turn_rate * dt
    )

    # Air-relative velocity
    air_vx = (
        HORIZONTAL_AIR_VELOCITY
        * math.cos(state.heading)
    )

    air_vy = (
        HORIZONTAL_AIR_VELOCITY
        * math.sin(state.heading)
    )

    # Ground velocity = air velocity + wind
    state.vx = (
        air_vx
        + wind_x
    )

    state.vy = (
        air_vy
        + wind_y
    )

    state.x += state.vx * dt
    state.y += state.vy * dt

    state.altitude -= (
        VERTICAL_DESCENT_VELOCITY
        * dt
    )

    if state.altitude < 0.0:
        state.altitude = 0.0


# ============================================================
# GUIDANCE MODEL
# ============================================================

def predict_candidate(
    x,
    y,
    altitude,
    heading,
    steering,
    horizon,
    wind_x,
    wind_y
):
    """
    Predict landing position for one candidate command.
    """

    sim_x = x
    sim_y = y
    sim_heading = heading

    predicted_turn_rate = (
        steering * MAX_TURN_RATE
    )

    remaining_time = (
        altitude /
        VERTICAL_DESCENT_VELOCITY
    )

    prediction_time = min(
        horizon,
        remaining_time
    )

    steps = max(
        1,
        int(
            prediction_time / 0.2
        )
    )

    dt = (
        prediction_time /
        steps
    )

    for _ in range(steps):

        sim_heading = wrap_angle(
            sim_heading
            + predicted_turn_rate * dt
        )

        vx = (
            HORIZONTAL_AIR_VELOCITY
            * math.cos(sim_heading)
            + wind_x
        )

        vy = (
            HORIZONTAL_AIR_VELOCITY
            * math.sin(sim_heading)
            + wind_y
        )

        sim_x += vx * dt
        sim_y += vy * dt

    # Continue approximately straight after horizon
    remaining = (
        remaining_time
        - prediction_time
    )

    if remaining > 0:

        vx = (
            HORIZONTAL_AIR_VELOCITY
            * math.cos(sim_heading)
            + wind_x
        )

        vy = (
            HORIZONTAL_AIR_VELOCITY
            * math.sin(sim_heading)
            + wind_y
        )

        sim_x += vx * remaining
        sim_y += vy * remaining

    return (
        sim_x,
        sim_y,
        sim_heading
    )


# ============================================================
# GUIDANCE COMMAND SELECTION
# ============================================================

def select_guidance_command(
    estimated_x,
    estimated_y,
    estimated_altitude,
    estimated_heading,
    current_steering,
    wind_x,
    wind_y
):

    horizon = get_prediction_horizon(
        estimated_altitude
    )

    best_command = 0.0
    best_cost = float("inf")

    for command in STEERING_COMMANDS:

        px, py, ph = predict_candidate(
            estimated_x,
            estimated_y,
            estimated_altitude,
            estimated_heading,
            command,
            horizon,
            wind_x,
            wind_y
        )

        position_error = math.sqrt(
            (
                px - TARGET_X
            ) ** 2
            +
            (
                py - TARGET_Y
            ) ** 2
        )

        target_heading = math.atan2(
            TARGET_Y - estimated_y,
            TARGET_X - estimated_x
        )

        heading_error = abs(
            wrap_angle(
                ph - target_heading
            )
        )

        heading_error_normalized = (
            heading_error / math.pi
        )

        cost = (
            POSITION_WEIGHT
            * position_error
            +
            POSITION_WEIGHT
            * 10.0
            * heading_error_normalized
            +
            0.05
            * abs(
                command
                - current_steering
            )
        )

        if cost < best_cost:

            best_cost = cost
            best_command = command

    return best_command


# ============================================================
# SENSOR GENERATION
# ============================================================

def generate_gnss_measurement(
    true_state,
    stats
):
    """
    Generate GNSS measurement.

    Outliers are generated occasionally and then
    rejected by EKF innovation gating.
    """

    x = true_state.x
    y = true_state.y

    vx = true_state.vx
    vy = true_state.vy

    # Normal measurement
    measured_x = (
        x
        + rng.normal(
            0.0,
            GNSS_POSITION_NOISE
        )
    )

    measured_y = (
        y
        + rng.normal(
            0.0,
            GNSS_POSITION_NOISE
        )
    )

    measured_vx = (
        vx
        + rng.normal(
            0.0,
            GNSS_VELOCITY_NOISE
        )
    )

    measured_vy = (
        vy
        + rng.normal(
            0.0,
            GNSS_VELOCITY_NOISE
        )
    )

    # Generate artificial outlier
    if (
        rng.random()
        < GNSS_OUTLIER_PROBABILITY
    ):

        angle = rng.uniform(
            0.0,
            2.0 * math.pi
        )

        measured_x += (
            POSITION_OUTLIER_MAGNITUDE
            * math.cos(angle)
        )

        measured_y += (
            POSITION_OUTLIER_MAGNITUDE
            * math.sin(angle)
        )

        measured_vx += (
            VELOCITY_OUTLIER_MAGNITUDE
            * math.cos(angle)
        )

        measured_vy += (
            VELOCITY_OUTLIER_MAGNITUDE
            * math.sin(angle)
        )

        stats["generated_outliers"] += 1

    return np.array([
        measured_x,
        measured_y,
        measured_vx,
        measured_vy
    ])


def generate_baro_measurement(
    true_state
):

    return (
        true_state.altitude
        + rng.normal(
            0.0,
            BARO_ALTITUDE_NOISE
        )
    )


def generate_imu_measurement(
    true_state
):

    heading = (
        true_state.heading
        + rng.normal(
            0.0,
            IMU_HEADING_NOISE
        )
    )

    turn_rate = (
        true_state.turn_rate
        + rng.normal(
            0.0,
            IMU_TURN_RATE_NOISE
        )
    )

    return (
        wrap_angle(heading),
        turn_rate
    )


# ============================================================
# GNSS OUTAGE MODEL
# ============================================================

def gnss_available(
    time,
    outage_duration
):
    """
    Continuous GNSS outage.

    The outage begins at 70 seconds into flight.
    """

    OUTAGE_START = 70.0

    if outage_duration <= 0.0:
        return True

    outage_end = (
        OUTAGE_START
        + outage_duration
    )

    if (
        time >= OUTAGE_START
        and time < outage_end
    ):
        return False

    return True


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(
    outage_duration,
    wind_x,
    wind_y
):

    true_state = ParafoilState()

    ekf = EKF()

    time = 0.0

    next_guidance = 0.0
    next_gnss = 0.0
    next_baro = 0.0
    next_imu = 0.0

    current_steering = 0.0

    steering_history = []

    position_errors = []
    altitude_errors = []
    heading_errors = []

    outage_position_errors = []

    stats = {

        "gnss_available": 0,
        "gnss_accepted": 0,
        "gnss_rejected": 0,
        "gnss_dropouts": 0,
        "generated_outliers": 0
    }

    outage_active_time = 0.0

    previous_steering_sign = 0

    steering_reversals = 0

    while (
        time < MAX_TIME
        and true_state.altitude > 0.0
    ):

        # ----------------------------------------------------
        # TRUE DYNAMICS
        # ----------------------------------------------------

        propagate_true_state(
            true_state,
            current_steering,
            DT,
            wind_x,
            wind_y
        )

        # ----------------------------------------------------
        # EKF PREDICTION
        # ----------------------------------------------------

        ekf.predict(DT)

        # ----------------------------------------------------
        # IMU UPDATE
        # ----------------------------------------------------

        if time >= next_imu:

            (
                imu_heading,
                imu_turn_rate
            ) = generate_imu_measurement(
                true_state
            )

            ekf.update_imu(
                imu_heading,
                imu_turn_rate
            )

            next_imu += IMU_DT

        # ----------------------------------------------------
        # BAROMETER UPDATE
        # ----------------------------------------------------

        if time >= next_baro:

            baro_measurement = (
                generate_baro_measurement(
                    true_state
                )
            )

            ekf.update_baro(
                baro_measurement
            )

            next_baro += BARO_DT

        # ----------------------------------------------------
        # GNSS UPDATE
        # ----------------------------------------------------

        if time >= next_gnss:

            if gnss_available(
                time,
                outage_duration
            ):

                stats[
                    "gnss_available"
                ] += 1

                gnss_measurement = (
                    generate_gnss_measurement(
                        true_state,
                        stats
                    )
                )

                ekf.update_gnss(
                    gnss_measurement,
                    stats
                )

            else:

                stats[
                    "gnss_dropouts"
                ] += 1

                outage_active_time += (
                    GNSS_DT
                )

            next_gnss += GNSS_DT

        # ----------------------------------------------------
        # GUIDANCE
        # ----------------------------------------------------

        if time >= next_guidance:

            new_command = (
                select_guidance_command(
                    ekf.x[0],
                    ekf.x[1],
                    ekf.x[4],
                    ekf.x[5],
                    current_steering,
                    wind_x,
                    wind_y
                )
            )

            # Steering reversal detection
            old_sign = (
                0
                if abs(current_steering) < 1e-9
                else int(
                    math.copysign(
                        1,
                        current_steering
                    )
                )
            )

            new_sign = (
                0
                if abs(new_command) < 1e-9
                else int(
                    math.copysign(
                        1,
                        new_command
                    )
                )
            )

            if (
                old_sign != 0
                and new_sign != 0
                and old_sign != new_sign
            ):
                steering_reversals += 1

            current_steering = new_command

            steering_history.append(
                current_steering
            )

            next_guidance += (
                GUIDANCE_INTERVAL
            )

        # ----------------------------------------------------
        # ESTIMATION ERRORS
        # ----------------------------------------------------

        pos_error = math.sqrt(
            (
                ekf.x[0]
                - true_state.x
            ) ** 2
            +
            (
                ekf.x[1]
                - true_state.y
            ) ** 2
        )

        altitude_error = abs(
            ekf.x[4]
            - true_state.altitude
        )

        heading_error = abs(
            math.degrees(
                wrap_angle(
                    ekf.x[5]
                    - true_state.heading
                )
            )
        )

        position_errors.append(
            pos_error
        )

        altitude_errors.append(
            altitude_error
        )

        heading_errors.append(
            heading_error
        )

        # During outage
        if not gnss_available(
            time,
            outage_duration
        ):

            outage_position_errors.append(
                pos_error
            )

        time += DT

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    true_landing_x = true_state.x
    true_landing_y = true_state.y

    estimated_landing_x = ekf.x[0]
    estimated_landing_y = ekf.x[1]

    true_landing_error = math.sqrt(
        (
            true_landing_x
            - TARGET_X
        ) ** 2
        +
        (
            true_landing_y
            - TARGET_Y
        ) ** 2
    )

    estimated_landing_error = math.sqrt(
        (
            estimated_landing_x
            - TARGET_X
        ) ** 2
        +
        (
            estimated_landing_y
            - TARGET_Y
        ) ** 2
    )

    mean_position_error = np.mean(
        position_errors
    )

    rms_position_error = math.sqrt(
        np.mean(
            np.square(
                position_errors
            )
        )
    )

    max_position_error = np.max(
        position_errors
    )

    mean_altitude_error = np.mean(
        altitude_errors
    )

    rms_altitude_error = math.sqrt(
        np.mean(
            np.square(
                altitude_errors
            )
        )
    )

    max_altitude_error = np.max(
        altitude_errors
    )

    mean_heading_error = np.mean(
        heading_errors
    )

    rms_heading_error = math.sqrt(
        np.mean(
            np.square(
                heading_errors
            )
        )
    )

    max_heading_error = np.max(
        heading_errors
    )

    if len(outage_position_errors) > 0:

        outage_rms_position_error = math.sqrt(
            np.mean(
                np.square(
                    outage_position_errors
                )
            )
        )

        outage_max_position_error = np.max(
            outage_position_errors
        )

    else:

        outage_rms_position_error = 0.0
        outage_max_position_error = 0.0

    average_steering = (
        np.mean(
            np.abs(
                steering_history
            )
        )
        if steering_history
        else 0.0
    )

    return {

        "outage_duration":
            outage_duration,

        "true_landing_x":
            true_landing_x,

        "true_landing_y":
            true_landing_y,

        "estimated_landing_x":
            estimated_landing_x,

        "estimated_landing_y":
            estimated_landing_y,

        "true_landing_error":
            true_landing_error,

        "estimated_landing_error":
            estimated_landing_error,

        "flight_time":
            time,

        "mean_position_error":
            mean_position_error,

        "rms_position_error":
            rms_position_error,

        "max_position_error":
            max_position_error,

        "mean_altitude_error":
            mean_altitude_error,

        "rms_altitude_error":
            rms_altitude_error,

        "max_altitude_error":
            max_altitude_error,

        "mean_heading_error":
            mean_heading_error,

        "rms_heading_error":
            rms_heading_error,

        "max_heading_error":
            max_heading_error,

        "outage_rms_position_error":
            outage_rms_position_error,

        "outage_max_position_error":
            outage_max_position_error,

        "gnss_available":
            stats["gnss_available"],

        "gnss_accepted":
            stats["gnss_accepted"],

        "gnss_rejected":
            stats["gnss_rejected"],

        "gnss_dropouts":
            stats["gnss_dropouts"],

        "generated_outliers":
            stats["generated_outliers"],

        "outage_active_time":
            outage_active_time,

        "average_steering":
            average_steering,

        "steering_reversals":
            steering_reversals
    }


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 64)
print("VIRTUAL PARAFOIL V10.3")
print("GNSS FAILURE-DURATION ROBUSTNESS STUDY")
print("=" * 64)

print()

print("=" * 64)
print("PARAFOIL PARAMETERS")
print("=" * 64)

print(f"Area: {AREA} m^2")
print(f"Mass: {MASS} kg")
print(f"CL: {CL}")
print(f"CD: {CD}")
print(f"Airspeed: {AIRSPEED} m/s")
print(
    f"Horizontal air velocity: "
    f"{HORIZONTAL_AIR_VELOCITY} m/s"
)
print(
    f"Vertical descent velocity: "
    f"{VERTICAL_DESCENT_VELOCITY} m/s"
)
print(f"Glide ratio: {GLIDE_RATIO}")

print()

print("=" * 64)
print("TARGET")
print("=" * 64)

print(f"Target X: {TARGET_X} m")
print(f"Target Y: {TARGET_Y} m")
print(f"Tolerance: {TOLERANCE} m")

print()

print("=" * 64)
print("REFERENCE WIND")
print("=" * 64)

print(
    f"Wind speed: "
    f"{WIND_SPEED} m/s"
)

print(
    f"Wind direction: "
    f"{WIND_DIRECTION_DEG} degrees"
)

print(
    f"Wind X: "
    f"{WIND_X} m/s"
)

print(
    f"Wind Y: "
    f"{WIND_Y} m/s"
)

print()

print("=" * 64)
print("SENSOR MODEL")
print("=" * 64)

print(
    f"GNSS position noise: "
    f"{GNSS_POSITION_NOISE} m"
)

print(
    f"GNSS velocity noise: "
    f"{GNSS_VELOCITY_NOISE} m/s"
)

print(
    f"Barometer altitude noise: "
    f"{BARO_ALTITUDE_NOISE} m"
)

print(
    f"IMU heading noise: "
    f"{IMU_HEADING_NOISE_DEG} deg"
)

print(
    f"IMU turn-rate noise: "
    f"{IMU_TURN_RATE_NOISE_DEG} deg/s"
)

print(
    f"GNSS update rate: "
    f"{GNSS_RATE} Hz"
)

print(
    f"Barometer update rate: "
    f"{BARO_RATE} Hz"
)

print(
    f"IMU update rate: "
    f"{IMU_RATE} Hz"
)

print()

print("=" * 64)
print("GNSS FAILURE MODEL")
print("=" * 64)

print(
    "GNSS outlier rejection: ENABLED"
)

print(
    f"GNSS outlier probability: "
    f"{GNSS_OUTLIER_PROBABILITY}"
)

print(
    f"Position outlier magnitude: "
    f"{POSITION_OUTLIER_MAGNITUDE} m"
)

print(
    f"Velocity outlier magnitude: "
    f"{VELOCITY_OUTLIER_MAGNITUDE} m/s"
)

print(
    "Continuous GNSS outage start: "
    "70 s"
)

print()

print("=" * 64)
print("GUIDANCE")
print("=" * 64)

print(
    f"Guidance interval: "
    f"{GUIDANCE_INTERVAL} s"
)

print(
    f"Candidate commands: "
    f"{NUM_CANDIDATES}"
)

print(
    f"Maximum turn rate: "
    f"{MAX_TURN_RATE_DEG} deg/s"
)

print()

print("Adaptive prediction horizon:")

print(
    "Altitude > 400 m  -> 20 s"
)

print(
    "Altitude 200-400 m -> 15 s"
)

print(
    "Altitude 100-200 m -> 10 s"
)

print(
    "Altitude < 100 m  -> 5 s"
)

print()

# ============================================================
# TEST CASES
# ============================================================

OUTAGE_DURATIONS = [
    0.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0
]


# ============================================================
# RUN STUDY
# ============================================================

print("=" * 64)
print("V10.3 GNSS FAILURE-DURATION STUDY")
print("=" * 64)

print()

print(
    "Continuous GNSS outage begins at "
    "70 seconds."
)

print(
    "Testing outage durations:"
)

print(
    OUTAGE_DURATIONS
)

print()

results = []

for i, outage in enumerate(
    OUTAGE_DURATIONS,
    start=1
):

    print("-" * 64)

    print(
        f"Simulation {i} / "
        f"{len(OUTAGE_DURATIONS)}"
    )

    print(
        f"GNSS outage duration: "
        f"{outage:.1f} s"
    )

    result = run_simulation(
        outage,
        WIND_X,
        WIND_Y
    )

    results.append(result)

    print(
        f"True landing error: "
        f"{result['true_landing_error']:.3f} m"
    )

    print(
        f"Estimated landing error: "
        f"{result['estimated_landing_error']:.3f} m"
    )

    print(
        f"EKF RMS position error: "
        f"{result['rms_position_error']:.3f} m"
    )

    print(
        f"EKF RMS altitude error: "
        f"{result['rms_altitude_error']:.3f} m"
    )

    print(
        f"EKF RMS heading error: "
        f"{result['rms_heading_error']:.3f} deg"
    )

    print(
        f"Maximum position estimation error: "
        f"{result['max_position_error']:.3f} m"
    )

    print(
        f"Maximum outage position error: "
        f"{result['outage_max_position_error']:.3f} m"
    )

    print(
        f"GNSS dropouts: "
        f"{result['gnss_dropouts']}"
    )

    print(
        f"GNSS rejected outliers: "
        f"{result['gnss_rejected']}"
    )

    if (
        result["true_landing_error"]
        <= TOLERANCE
    ):

        print(
            "Landing status: "
            "WITHIN TOLERANCE"
        )

    else:

        print(
            "Landing status: "
            "OUTSIDE TOLERANCE"
        )


# ============================================================
# RESULTS TABLE
# ============================================================

print()
print()
print("=" * 64)
print("V10.3 GNSS FAILURE-DURATION RESULTS")
print("=" * 64)

print(
    "Outage(s)   Landing Error   EKF RMS Pos   "
    "Outage RMS Pos   Max Outage Error   Status"
)

print("-" * 64)

for r in results:

    if (
        r["true_landing_error"]
        <= TOLERANCE
    ):
        status = "PASS"
    else:
        status = "FAIL"

    print(
        f"{r['outage_duration']:8.1f}   "
        f"{r['true_landing_error']:14.3f}   "
        f"{r['rms_position_error']:12.3f}   "
        f"{r['outage_rms_position_error']:15.3f}   "
        f"{r['outage_max_position_error']:16.3f}   "
        f"{status}"
    )


# ============================================================
# BEST / WORST LANDING CASE
# ============================================================

best_landing = min(
    results,
    key=lambda r:
    r["true_landing_error"]
)

worst_landing = max(
    results,
    key=lambda r:
    r["true_landing_error"]
)


# ============================================================
# LONGEST SUCCESSFUL OUTAGE
# ============================================================

successful_results = [
    r for r in results
    if r["true_landing_error"]
    <= TOLERANCE
]

if successful_results:

    longest_successful = max(
        successful_results,
        key=lambda r:
        r["outage_duration"]
    )

else:

    longest_successful = None


# ============================================================
# FIRST FAILURE
# ============================================================

failed_results = [
    r for r in results
    if r["true_landing_error"]
    > TOLERANCE
]

if failed_results:

    first_failure = min(
        failed_results,
        key=lambda r:
        r["outage_duration"]
    )

else:

    first_failure = None


# ============================================================
# BEST LANDING
# ============================================================

print()
print("=" * 64)
print("BEST LANDING CASE")
print("=" * 64)

print(
    f"GNSS outage duration: "
    f"{best_landing['outage_duration']:.1f} s"
)

print(
    f"Landing X: "
    f"{best_landing['true_landing_x']:.3f} m"
)

print(
    f"Landing Y: "
    f"{best_landing['true_landing_y']:.3f} m"
)

print(
    f"Landing error: "
    f"{best_landing['true_landing_error']:.3f} m"
)


# ============================================================
# WORST LANDING
# ============================================================

print()
print("=" * 64)
print("WORST LANDING CASE")
print("=" * 64)

print(
    f"GNSS outage duration: "
    f"{worst_landing['outage_duration']:.1f} s"
)

print(
    f"Landing X: "
    f"{worst_landing['true_landing_x']:.3f} m"
)

print(
    f"Landing Y: "
    f"{worst_landing['true_landing_y']:.3f} m"
)

print(
    f"Landing error: "
    f"{worst_landing['true_landing_error']:.3f} m"
)


# ============================================================
# LONGEST SUCCESSFUL OUTAGE
# ============================================================

print()
print("=" * 64)
print("GNSS OUTAGE ROBUSTNESS")
print("=" * 64)

if longest_successful is not None:

    print(
        "Longest tested GNSS outage with "
        "landing inside tolerance:"
    )

    print(
        f"{longest_successful['outage_duration']:.1f} s"
    )

    print(
        f"Landing error: "
        f"{longest_successful['true_landing_error']:.3f} m"
    )

else:

    print(
        "No tested GNSS outage duration "
        "maintained landing within tolerance."
    )


# ============================================================
# FIRST FAILURE
# ============================================================

print()

if first_failure is not None:

    print(
        "First tested outage duration "
        "causing landing failure:"
    )

    print(
        f"{first_failure['outage_duration']:.1f} s"
    )

    print(
        f"Landing error: "
        f"{first_failure['true_landing_error']:.3f} m"
    )

else:

    print(
        "No tested outage caused "
        "landing failure."
    )


# ============================================================
# ESTIMATION PERFORMANCE
# ============================================================

best_estimation = min(
    results,
    key=lambda r:
    r["rms_position_error"]
)

worst_estimation = max(
    results,
    key=lambda r:
    r["rms_position_error"]
)

print()
print("=" * 64)
print("EKF ESTIMATION PERFORMANCE")
print("=" * 64)

print(
    "Best RMS position estimation:"
)

print(
    f"GNSS outage: "
    f"{best_estimation['outage_duration']:.1f} s"
)

print(
    f"RMS position error: "
    f"{best_estimation['rms_position_error']:.3f} m"
)

print()

print(
    "Worst RMS position estimation:"
)

print(
    f"GNSS outage: "
    f"{worst_estimation['outage_duration']:.1f} s"
)

print(
    f"RMS position error: "
    f"{worst_estimation['rms_position_error']:.3f} m"
)


# ============================================================
# NAVIGATION SYSTEM ASSESSMENT
# ============================================================

print()
print("=" * 64)
print("V10.3 NAVIGATION SYSTEM ASSESSMENT")
print("=" * 64)

print(
    f"Raw GNSS position noise: "
    f"{GNSS_POSITION_NOISE} m"
)

print(
    f"Best EKF RMS position error: "
    f"{best_estimation['rms_position_error']:.3f} m"
)

if (
    best_estimation["rms_position_error"]
    < GNSS_POSITION_NOISE
):

    print(
        "EKF STATUS: ESTIMATION IMPROVED"
    )

else:

    print(
        "EKF STATUS: ESTIMATION NOT IMPROVED"
    )

print(
    "GNSS OUTLIER REJECTION: ACTIVE"
)

print(
    "GNSS FAILURE HANDLING: ACTIVE"
)


# ============================================================
# FINAL DIAGNOSTIC
# ============================================================

print()
print("=" * 64)
print("V10.3 DIAGNOSTIC SUMMARY")
print("=" * 64)

print()

print(
    "The simulation evaluates the ability of the "
    "EKF to maintain navigation state during "
    "continuous GNSS outages."
)

print()

if longest_successful is not None:

    print(
        f"Maximum tested successful outage: "
        f"{longest_successful['outage_duration']:.1f} s"
    )

if first_failure is not None:

    print(
        f"First tested failure outage: "
        f"{first_failure['outage_duration']:.1f} s"
    )

print()

print(
    "During GNSS outage the EKF propagates the "
    "state using the process model and IMU/barometer "
    "measurements."
)

print()

print(
    "When GNSS becomes available again, the EKF "
    "uses the GNSS measurement to correct accumulated "
    "state drift."
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 64)
print("V10.3 SIMULATION COMPLETE")
print("=" * 64)

print(
    f"Reference wind: "
    f"{WIND_SPEED} m/s"
)

print(
    f"Reference direction: "
    f"{WIND_DIRECTION_DEG} degrees"
)

print(
    f"GNSS rate: "
    f"{GNSS_RATE} Hz"
)

print(
    f"Barometer rate: "
    f"{BARO_RATE} Hz"
)

print(
    f"IMU rate: "
    f"{IMU_RATE} Hz"
)

print(
    f"GNSS outage durations tested: "
    f"{OUTAGE_DURATIONS}"
)

print(
    f"Best landing error: "
    f"{best_landing['true_landing_error']:.3f} m"
)

print(
    f"Worst landing error: "
    f"{worst_landing['true_landing_error']:.3f} m"
)

if longest_successful is not None:

    print(
        f"Longest successful GNSS outage: "
        f"{longest_successful['outage_duration']:.1f} s"
    )

else:

    print(
        "Longest successful GNSS outage: "
        "None"
    )

print(
    "GNSS outlier rejection: ACTIVE"
)

print(
    "EKF failure-duration robustness study: COMPLETE"
)

print("=" * 64)

print()
print(
    "NEXT DEVELOPMENT STEP:"
)

print(
    "V10.4 -> Multi-sensor failure / "
    "GNSS + IMU + barometer robustness"
)

print("=" * 64)