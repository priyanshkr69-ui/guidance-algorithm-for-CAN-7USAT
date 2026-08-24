"""
================================================================
VIRTUAL PARAFOIL V10.4
MULTI-SENSOR FAILURE ROBUSTNESS
GNSS + IMU + BAROMETER FAILURE STUDY
================================================================

Purpose:
    Evaluate parafoil navigation and guidance robustness when
    GNSS, IMU and barometer measurements experience failures.

Features:
    1. Parafoil flight dynamics
    2. Wind model
    3. Adaptive prediction horizon
    4. Candidate steering guidance
    5. GNSS noise
    6. GNSS outlier rejection
    7. GNSS continuous outage
    8. IMU outage/degradation
    9. Barometer outage/degradation
    10. EKF state estimation
    11. Multi-sensor failure scenarios
    12. Landing performance comparison
================================================================
"""

import math
import random
import numpy as np


# ================================================================
# RANDOM SEED
# ================================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ================================================================
# PARAFOIL PARAMETERS
# ================================================================

AREA = 0.96
MASS = 1.0

CL = 0.40
CD = 0.25

AIR_DENSITY = 1.225
G = 9.81

AIRSPEED = 7.0132343805211725

HORIZONTAL_AIR_SPEED = 5.947210860272128
VERTICAL_DESCENT_SPEED = 3.71700678767008

GLIDE_RATIO = (
    HORIZONTAL_AIR_SPEED /
    VERTICAL_DESCENT_SPEED
)

START_ALTITUDE = 600.0

INITIAL_X = 0.0
INITIAL_Y = 0.0


# ================================================================
# TARGET
# ================================================================

TARGET_X = 500.0
TARGET_Y = 200.0

TARGET_TOLERANCE = 20.0


# ================================================================
# WIND
# ================================================================

REFERENCE_WIND_SPEED = 3.0
REFERENCE_WIND_DIRECTION = 0.0

REFERENCE_WIND_X = (
    REFERENCE_WIND_SPEED *
    math.cos(math.radians(REFERENCE_WIND_DIRECTION))
)

REFERENCE_WIND_Y = (
    REFERENCE_WIND_SPEED *
    math.sin(math.radians(REFERENCE_WIND_DIRECTION))
)


# ================================================================
# SIMULATION
# ================================================================

DT = 0.02

MAX_TIME = (
    START_ALTITUDE /
    VERTICAL_DESCENT_SPEED
)

GUIDANCE_INTERVAL = 2.0

MAX_STEERING = 1.0

NUM_CANDIDATES = 21

STEERING_COMMANDS = np.linspace(
    -1.0,
    1.0,
    NUM_CANDIDATES
)

MAX_TURN_RATE_DEG = 15.0
MAX_TURN_RATE = math.radians(MAX_TURN_RATE_DEG)


# ================================================================
# ADAPTIVE HORIZON
# ================================================================

def get_prediction_horizon(altitude):

    if altitude > 400.0:
        return 20.0

    elif altitude > 200.0:
        return 15.0

    elif altitude > 100.0:
        return 10.0

    else:
        return 5.0


# ================================================================
# SENSOR PARAMETERS
# ================================================================

GNSS_POSITION_NOISE = 3.0
GNSS_VELOCITY_NOISE = 0.3

BARO_ALTITUDE_NOISE = 2.0

IMU_HEADING_NOISE_DEG = 2.0
IMU_TURN_RATE_NOISE_DEG = 0.5

GNSS_RATE = 5.0
BARO_RATE = 10.0
IMU_RATE = 50.0


# ================================================================
# FAILURE PARAMETERS
# ================================================================

GNSS_OUTLIER_PROBABILITY = 0.05

GNSS_POSITION_OUTLIER = 40.0
GNSS_VELOCITY_OUTLIER = 5.0

GNSS_OUTAGE_START = 70.0

IMU_OUTAGE_START = 90.0
IMU_OUTAGE_DURATION = 10.0

BARO_OUTAGE_START = 120.0
BARO_OUTAGE_DURATION = 10.0


# ================================================================
# FAILURE SCENARIOS
# ================================================================

SCENARIOS = [

    {
        "name": "Nominal",
        "gnss": True,
        "imu": True,
        "baro": True,
    },

    {
        "name": "GNSS outage",
        "gnss": False,
        "imu": True,
        "baro": True,
    },

    {
        "name": "IMU outage",
        "gnss": True,
        "imu": False,
        "baro": True,
    },

    {
        "name": "Barometer outage",
        "gnss": True,
        "imu": True,
        "baro": False,
    },

    {
        "name": "GNSS + IMU outage",
        "gnss": False,
        "imu": False,
        "baro": True,
    },

    {
        "name": "GNSS + Barometer outage",
        "gnss": False,
        "imu": True,
        "baro": False,
    },

    {
        "name": "IMU + Barometer outage",
        "gnss": True,
        "imu": False,
        "baro": False,
    },

    {
        "name": "ALL SENSOR FAILURE",
        "gnss": False,
        "imu": False,
        "baro": False,
    },
]


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def wrap_angle(angle):

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def distance(x1, y1, x2, y2):

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def clamp(value, low, high):

    return max(low, min(high, value))


# ================================================================
# PARAFOIL DYNAMICS
# ================================================================

def propagate_true_state(
        x,
        y,
        altitude,
        heading,
        steering,
        wind_x,
        wind_y,
        dt):

    steering = clamp(
        steering,
        -1.0,
        1.0
    )

    # Maximum heading change
    heading_rate = (
        MAX_TURN_RATE *
        steering
    )

    new_heading = wrap_angle(
        heading +
        heading_rate * dt
    )

    # Air-relative velocity
    air_vx = (
        HORIZONTAL_AIR_SPEED *
        math.cos(new_heading)
    )

    air_vy = (
        HORIZONTAL_AIR_SPEED *
        math.sin(new_heading)
    )

    # Ground velocity
    ground_vx = air_vx + wind_x
    ground_vy = air_vy + wind_y

    new_x = (
        x +
        ground_vx * dt
    )

    new_y = (
        y +
        ground_vy * dt
    )

    new_altitude = max(
        0.0,
        altitude -
        VERTICAL_DESCENT_SPEED * dt
    )

    return (
        new_x,
        new_y,
        new_altitude,
        new_heading,
        ground_vx,
        ground_vy
    )


# ================================================================
# EKF
# ================================================================

class ParafoilEKF:

    """
    State:

        x
        y
        altitude
        vx
        vy
        vertical_velocity
        heading
        turn_rate

    """

    def __init__(
            self,
            x0,
            y0,
            altitude0,
            heading0):

        self.x = np.array([
            x0,
            y0,
            altitude0,
            HORIZONTAL_AIR_SPEED,
            0.0,
            VERTICAL_DESCENT_SPEED,
            heading0,
            0.0
        ], dtype=float)

        self.P = np.diag([
            25.0,
            25.0,
            16.0,
            4.0,
            4.0,
            1.0,
            math.radians(10.0) ** 2,
            math.radians(5.0) ** 2
        ])

        self.Q = np.diag([
            0.03,
            0.03,
            0.05,
            0.15,
            0.15,
            0.05,
            math.radians(0.2) ** 2,
            math.radians(0.5) ** 2
        ])

    # ------------------------------------------------------------

    def predict(
            self,
            dt,
            steering,
            wind_x,
            wind_y):

        x = self.x

        heading = x[6]

        turn_rate = (
            MAX_TURN_RATE *
            steering
        )

        heading_new = wrap_angle(
            heading +
            turn_rate * dt
        )

        vx = (
            HORIZONTAL_AIR_SPEED *
            math.cos(heading_new)
            +
            wind_x
        )

        vy = (
            HORIZONTAL_AIR_SPEED *
            math.sin(heading_new)
            +
            wind_y
        )

        altitude = max(
            0.0,
            x[2] -
            VERTICAL_DESCENT_SPEED * dt
        )

        self.x[0] += vx * dt
        self.x[1] += vy * dt
        self.x[2] = altitude

        self.x[3] = vx
        self.x[4] = vy

        self.x[5] = VERTICAL_DESCENT_SPEED

        self.x[6] = heading_new

        self.x[7] = turn_rate

        # Simplified covariance propagation
        self.P += self.Q * dt

        # Numerical stabilization
        self.P = (
            self.P +
            self.P.T
        ) / 2.0

    # ------------------------------------------------------------

    def update_gnss(
            self,
            px,
            py,
            vx,
            vy):

        z = np.array([
            px,
            py,
            vx,
            vy
        ])

        h = np.array([
            self.x[0],
            self.x[1],
            self.x[3],
            self.x[4]
        ])

        residual = z - h

        H = np.zeros((4, 8))

        H[0, 0] = 1.0
        H[1, 1] = 1.0

        H[2, 3] = 1.0
        H[3, 4] = 1.0

        R = np.diag([
            GNSS_POSITION_NOISE ** 2,
            GNSS_POSITION_NOISE ** 2,
            GNSS_VELOCITY_NOISE ** 2,
            GNSS_VELOCITY_NOISE ** 2
        ])

        S = (
            H @ self.P @ H.T
            +
            R
        )

        # Mahalanobis distance
        try:

            mahalanobis = (
                residual.T @
                np.linalg.inv(S) @
                residual
            )

        except np.linalg.LinAlgError:

            return False

        # Outlier rejection
        if mahalanobis > 16.0:

            return False

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x = (
            self.x +
            K @ residual
        )

        I = np.eye(8)

        self.P = (
            (I - K @ H) @
            self.P
        )

        self.P = (
            self.P +
            self.P.T
        ) / 2.0

        self.x[6] = wrap_angle(
            self.x[6]
        )

        return True

    # ------------------------------------------------------------

    def update_barometer(
            self,
            altitude):

        H = np.zeros((1, 8))

        H[0, 2] = 1.0

        residual = np.array([
            altitude -
            self.x[2]
        ])

        R = np.array([
            [BARO_ALTITUDE_NOISE ** 2]
        ])

        S = (
            H @ self.P @ H.T
            +
            R
        )

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x = (
            self.x +
            (K @ residual)
        )

        I = np.eye(8)

        self.P = (
            (I - K @ H) @
            self.P
        )

        self.P = (
            self.P +
            self.P.T
        ) / 2.0

        return True

    # ------------------------------------------------------------

    def update_imu(
            self,
            heading,
            turn_rate):

        H = np.zeros((2, 8))

        H[0, 6] = 1.0
        H[1, 7] = 1.0

        residual = np.array([
            wrap_angle(
                heading -
                self.x[6]
            ),
            turn_rate -
            self.x[7]
        ])

        R = np.diag([
            math.radians(
                IMU_HEADING_NOISE_DEG
            ) ** 2,

            math.radians(
                IMU_TURN_RATE_NOISE_DEG
            ) ** 2
        ])

        S = (
            H @ self.P @ H.T
            +
            R
        )

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x = (
            self.x +
            K @ residual
        )

        self.x[6] = wrap_angle(
            self.x[6]
        )

        I = np.eye(8)

        self.P = (
            (I - K @ H) @
            self.P
        )

        self.P = (
            self.P +
            self.P.T
        ) / 2.0

        return True


# ================================================================
# SENSOR GENERATION
# ================================================================

def generate_gnss(
        true_x,
        true_y,
        true_vx,
        true_vy):

    px = (
        true_x +
        np.random.normal(
            0.0,
            GNSS_POSITION_NOISE
        )
    )

    py = (
        true_y +
        np.random.normal(
            0.0,
            GNSS_POSITION_NOISE
        )
    )

    vx = (
        true_vx +
        np.random.normal(
            0.0,
            GNSS_VELOCITY_NOISE
        )
    )

    vy = (
        true_vy +
        np.random.normal(
            0.0,
            GNSS_VELOCITY_NOISE
        )
    )

    # Artificial outlier
    if random.random() < GNSS_OUTLIER_PROBABILITY:

        px += (
            random.choice([-1, 1]) *
            GNSS_POSITION_OUTLIER
        )

        py += (
            random.choice([-1, 1]) *
            GNSS_POSITION_OUTLIER
        )

        vx += (
            random.choice([-1, 1]) *
            GNSS_VELOCITY_OUTLIER
        )

        vy += (
            random.choice([-1, 1]) *
            GNSS_VELOCITY_OUTLIER
        )

    return px, py, vx, vy


# ================================================================
# GUIDANCE COST
# ================================================================

def predicted_position(
        x,
        y,
        heading,
        altitude,
        steering,
        horizon,
        wind_x,
        wind_y):

    px = x
    py = y

    ph = heading

    remaining = horizon

    step = 0.5

    while remaining > 0.0:

        dt = min(
            step,
            remaining
        )

        turn_rate = (
            MAX_TURN_RATE *
            steering
        )

        ph = wrap_angle(
            ph +
            turn_rate * dt
        )

        vx = (
            HORIZONTAL_AIR_SPEED *
            math.cos(ph)
            +
            wind_x
        )

        vy = (
            HORIZONTAL_AIR_SPEED *
            math.sin(ph)
            +
            wind_y
        )

        px += vx * dt
        py += vy * dt

        remaining -= dt

    return px, py


# ================================================================
# GUIDANCE COMMAND
# ================================================================

def select_guidance_command(
        state,
        wind_x,
        wind_y):

    x = state[0]
    y = state[1]
    altitude = state[2]
    heading = state[6]

    horizon = get_prediction_horizon(
        altitude
    )

    best_command = 0.0

    best_cost = float("inf")

    distance_now = distance(
        x,
        y,
        TARGET_X,
        TARGET_Y
    )

    for command in STEERING_COMMANDS:

        px, py = predicted_position(
            x,
            y,
            heading,
            altitude,
            command,
            horizon,
            wind_x,
            wind_y
        )

        position_error = distance(
            px,
            py,
            TARGET_X,
            TARGET_Y
        )

        # Heading toward target
        desired_heading = math.atan2(
            TARGET_Y - y,
            TARGET_X - x
        )

        predicted_heading = wrap_angle(
            heading +
            MAX_TURN_RATE *
            command *
            horizon
        )

        heading_error = abs(
            wrap_angle(
                predicted_heading -
                desired_heading
            )
        )

        cost = (
            0.70 *
            position_error
            +
            0.30 *
            heading_error *
            50.0
        )

        # Penalize excessive steering
        cost += (
            2.0 *
            abs(command)
        )

        # Near target, reduce unnecessary oscillation
        if distance_now < 100.0:

            cost += (
                5.0 *
                abs(command)
            )

        if cost < best_cost:

            best_cost = cost
            best_command = command

    return best_command


# ================================================================
# FAILURE CHECK
# ================================================================

def sensor_available(
        sensor_name,
        time,
        scenario):

    if sensor_name == "gnss":

        if not scenario["gnss"]:
            return False

        if (
            GNSS_OUTAGE_START <=
            time <
            GNSS_OUTAGE_START + 30.0
        ):
            return False

        return True

    if sensor_name == "imu":

        if not scenario["imu"]:
            return False

        if (
            IMU_OUTAGE_START <=
            time <
            IMU_OUTAGE_START +
            IMU_OUTAGE_DURATION
        ):
            return False

        return True

    if sensor_name == "baro":

        if not scenario["baro"]:
            return False

        if (
            BARO_OUTAGE_START <=
            time <
            BARO_OUTAGE_START +
            BARO_OUTAGE_DURATION
        ):
            return False

        return True

    return False


# ================================================================
# RUN SIMULATION
# ================================================================

def run_simulation(
        scenario,
        wind_x,
        wind_y):

    # True state
    x = INITIAL_X
    y = INITIAL_Y

    altitude = START_ALTITUDE

    heading = math.atan2(
        TARGET_Y,
        TARGET_X
    )

    # EKF
    ekf = ParafoilEKF(
        x,
        y,
        altitude,
        heading
    )

    time = 0.0

    next_guidance = 0.0

    steering = 0.0

    previous_steering = 0.0

    steering_reversals = 0

    steering_sum = 0.0
    steering_count = 0

    # Error storage
    position_errors = []
    altitude_errors = []
    heading_errors = []

    outage_position_errors = []

    # Sensor statistics
    gnss_available_count = 0
    gnss_accepted = 0
    gnss_rejected = 0

    gnss_dropouts = 0
    imu_dropouts = 0
    baro_dropouts = 0

    # Main simulation
    while (
        time < MAX_TIME
        and altitude > 0.0
    ):

        # --------------------------------------------------------
        # TRUE DYNAMICS
        # --------------------------------------------------------

        (
            x,
            y,
            altitude,
            heading,
            true_vx,
            true_vy
        ) = propagate_true_state(
            x,
            y,
            altitude,
            heading,
            steering,
            wind_x,
            wind_y,
            DT
        )

        # --------------------------------------------------------
        # EKF PREDICTION
        # --------------------------------------------------------

        ekf.predict(
            DT,
            steering,
            wind_x,
            wind_y
        )

        # --------------------------------------------------------
        # GNSS
        # --------------------------------------------------------

        gnss_period = 1.0 / GNSS_RATE

        gnss_phase = (
            time %
            gnss_period
        )

        if gnss_phase < DT:

            if sensor_available(
                "gnss",
                time,
                scenario
            ):

                gnss_available_count += 1

                (
                    px,
                    py,
                    gvx,
                    gvy
                ) = generate_gnss(
                    x,
                    y,
                    true_vx,
                    true_vy
                )

                accepted = (
                    ekf.update_gnss(
                        px,
                        py,
                        gvx,
                        gvy
                    )
                )

                if accepted:

                    gnss_accepted += 1

                else:

                    gnss_rejected += 1

            else:

                gnss_dropouts += 1

        # --------------------------------------------------------
        # BAROMETER
        # --------------------------------------------------------

        baro_period = 1.0 / BARO_RATE

        if (
            time %
            baro_period
        ) < DT:

            if sensor_available(
                "baro",
                time,
                scenario
            ):

                measured_altitude = (
                    altitude +
                    np.random.normal(
                        0.0,
                        BARO_ALTITUDE_NOISE
                    )
                )

                ekf.update_barometer(
                    measured_altitude
                )

            else:

                baro_dropouts += 1

        # --------------------------------------------------------
        # IMU
        # --------------------------------------------------------

        imu_period = 1.0 / IMU_RATE

        if (
            time %
            imu_period
        ) < DT:

            if sensor_available(
                "imu",
                time,
                scenario
            ):

                true_turn_rate = (
                    MAX_TURN_RATE *
                    steering
                )

                measured_heading = (
                    heading +
                    np.random.normal(
                        0.0,
                        math.radians(
                            IMU_HEADING_NOISE_DEG
                        )
                    )
                )

                measured_turn_rate = (
                    true_turn_rate +
                    np.random.normal(
                        0.0,
                        math.radians(
                            IMU_TURN_RATE_NOISE_DEG
                        )
                    )
                )

                ekf.update_imu(
                    measured_heading,
                    measured_turn_rate
                )

            else:

                imu_dropouts += 1

        # --------------------------------------------------------
        # STATE ESTIMATION ERROR
        # --------------------------------------------------------

        pos_error = distance(
            ekf.x[0],
            ekf.x[1],
            x,
            y
        )

        altitude_error = abs(
            ekf.x[2] -
            altitude
        )

        heading_error = abs(
            math.degrees(
                wrap_angle(
                    ekf.x[6] -
                    heading
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

        # During combined failure periods
        if (
            not sensor_available(
                "gnss",
                time,
                scenario
            )
        ):

            outage_position_errors.append(
                pos_error
            )

        # --------------------------------------------------------
        # GUIDANCE
        # --------------------------------------------------------

        if time >= next_guidance:

            steering = (
                select_guidance_command(
                    ekf.x,
                    wind_x,
                    wind_y
                )
            )

            if (
                steering *
                previous_steering
                < 0.0
                and
                abs(steering) > 0.05
                and
                abs(previous_steering) > 0.05
            ):

                steering_reversals += 1

            previous_steering = steering

            next_guidance += (
                GUIDANCE_INTERVAL
            )

        steering_sum += abs(
            steering
        )

        steering_count += 1

        time += DT

    # ============================================================
    # FINAL RESULTS
    # ============================================================

    true_landing_error = distance(
        x,
        y,
        TARGET_X,
        TARGET_Y
    )

    estimated_landing_error = distance(
        ekf.x[0],
        ekf.x[1],
        TARGET_X,
        TARGET_Y
    )

    position_errors = np.array(
        position_errors
    )

    altitude_errors = np.array(
        altitude_errors
    )

    heading_errors = np.array(
        heading_errors
    )

    if len(outage_position_errors) > 0:

        outage_position_errors = np.array(
            outage_position_errors
        )

        outage_rms = math.sqrt(
            np.mean(
                outage_position_errors ** 2
            )
        )

        outage_max = np.max(
            outage_position_errors
        )

    else:

        outage_rms = 0.0
        outage_max = 0.0

    return {

        "scenario":
            scenario["name"],

        "landing_x":
            x,

        "landing_y":
            y,

        "estimated_x":
            ekf.x[0],

        "estimated_y":
            ekf.x[1],

        "true_landing_error":
            true_landing_error,

        "estimated_landing_error":
            estimated_landing_error,

        "rms_position":
            math.sqrt(
                np.mean(
                    position_errors ** 2
                )
            ),

        "rms_altitude":
            math.sqrt(
                np.mean(
                    altitude_errors ** 2
                )
            ),

        "rms_heading":
            math.sqrt(
                np.mean(
                    heading_errors ** 2
                )
            ),

        "max_position":
            np.max(
                position_errors
            ),

        "outage_rms":
            outage_rms,

        "outage_max":
            outage_max,

        "gnss_available":
            gnss_available_count,

        "gnss_accepted":
            gnss_accepted,

        "gnss_rejected":
            gnss_rejected,

        "gnss_dropouts":
            gnss_dropouts,

        "imu_dropouts":
            imu_dropouts,

        "baro_dropouts":
            baro_dropouts,

        "average_steering":
            steering_sum /
            max(
                1,
                steering_count
            ),

        "steering_reversals":
            steering_reversals,

        "flight_time":
            time
    }


# ================================================================
# PRINT HEADER
# ================================================================

print()
print("=" * 64)
print("VIRTUAL PARAFOIL V10.4")
print("MULTI-SENSOR FAILURE ROBUSTNESS")
print("GNSS + IMU + BAROMETER FAILURE STUDY")
print("=" * 64)

print()
print("=" * 64)
print("PARAFOIL PARAMETERS")
print("=" * 64)

print("Area:", AREA, "m^2")
print("Mass:", MASS, "kg")
print("CL:", CL)
print("CD:", CD)
print("Airspeed:", AIRSPEED, "m/s")
print(
    "Horizontal air velocity:",
    HORIZONTAL_AIR_SPEED,
    "m/s"
)
print(
    "Vertical descent velocity:",
    VERTICAL_DESCENT_SPEED,
    "m/s"
)
print("Glide ratio:", GLIDE_RATIO)

print()
print("=" * 64)
print("TARGET")
print("=" * 64)

print(
    "Target X:",
    TARGET_X,
    "m"
)

print(
    "Target Y:",
    TARGET_Y,
    "m"
)

print(
    "Tolerance:",
    TARGET_TOLERANCE,
    "m"
)

print()
print("=" * 64)
print("REFERENCE WIND")
print("=" * 64)

print(
    "Wind speed:",
    REFERENCE_WIND_SPEED,
    "m/s"
)

print(
    "Wind direction:",
    REFERENCE_WIND_DIRECTION,
    "degrees"
)

print(
    "Wind X:",
    REFERENCE_WIND_X,
    "m/s"
)

print(
    "Wind Y:",
    REFERENCE_WIND_Y,
    "m/s"
)

print()
print("=" * 64)
print("SENSOR MODEL")
print("=" * 64)

print(
    "GNSS position noise:",
    GNSS_POSITION_NOISE,
    "m"
)

print(
    "GNSS velocity noise:",
    GNSS_VELOCITY_NOISE,
    "m/s"
)

print(
    "Barometer altitude noise:",
    BARO_ALTITUDE_NOISE,
    "m"
)

print(
    "IMU heading noise:",
    IMU_HEADING_NOISE_DEG,
    "deg"
)

print(
    "IMU turn-rate noise:",
    IMU_TURN_RATE_NOISE_DEG,
    "deg/s"
)

print(
    "GNSS update rate:",
    GNSS_RATE,
    "Hz"
)

print(
    "Barometer update rate:",
    BARO_RATE,
    "Hz"
)

print(
    "IMU update rate:",
    IMU_RATE,
    "Hz"
)

print()
print("=" * 64)
print("FAILURE MODEL")
print("=" * 64)

print(
    "GNSS outlier rejection: ENABLED"
)

print(
    "GNSS outlier probability:",
    GNSS_OUTLIER_PROBABILITY
)

print(
    "GNSS position outlier:",
    GNSS_POSITION_OUTLIER,
    "m"
)

print(
    "GNSS velocity outlier:",
    GNSS_VELOCITY_OUTLIER,
    "m/s"
)

print(
    "GNSS outage start:",
    GNSS_OUTAGE_START,
    "s"
)

print(
    "IMU degradation period:",
    IMU_OUTAGE_START,
    "-",
    IMU_OUTAGE_START +
    IMU_OUTAGE_DURATION,
    "s"
)

print(
    "Barometer degradation period:",
    BARO_OUTAGE_START,
    "-",
    BARO_OUTAGE_START +
    BARO_OUTAGE_DURATION,
    "s"
)

print()
print("=" * 64)
print("GUIDANCE")
print("=" * 64)

print(
    "Guidance interval:",
    GUIDANCE_INTERVAL,
    "s"
)

print(
    "Candidate commands:",
    NUM_CANDIDATES
)

print(
    "Maximum turn rate:",
    MAX_TURN_RATE_DEG,
    "deg/s"
)

print()
print(
    "Adaptive prediction horizon:"
)

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

# ================================================================
# RUN ALL SCENARIOS
# ================================================================

print()
print("=" * 64)
print("STARTING V10.4 MULTI-SENSOR FAILURE STUDY")
print("=" * 64)

results = []

for i, scenario in enumerate(
        SCENARIOS,
        start=1):

    print()
    print("-" * 64)

    print(
        "Scenario",
        i,
        "/",
        len(SCENARIOS)
    )

    print(
        "Failure mode:",
        scenario["name"]
    )

    simulation = run_simulation(
        scenario,
        REFERENCE_WIND_X,
        REFERENCE_WIND_Y
    )

    results.append(
        simulation
    )

    status = (
        "WITHIN TOLERANCE"
        if simulation[
            "true_landing_error"
        ] <= TARGET_TOLERANCE
        else
        "OUTSIDE TOLERANCE"
    )

    print(
        "True landing error:",
        f"{simulation['true_landing_error']:.3f}",
        "m"
    )

    print(
        "Estimated landing error:",
        f"{simulation['estimated_landing_error']:.3f}",
        "m"
    )

    print(
        "EKF RMS position:",
        f"{simulation['rms_position']:.3f}",
        "m"
    )

    print(
        "EKF RMS altitude:",
        f"{simulation['rms_altitude']:.3f}",
        "m"
    )

    print(
        "EKF RMS heading:",
        f"{simulation['rms_heading']:.3f}",
        "deg"
    )

    print(
        "Maximum position error:",
        f"{simulation['max_position']:.3f}",
        "m"
    )

    print(
        "GNSS dropouts:",
        simulation["gnss_dropouts"]
    )

    print(
        "GNSS rejected:",
        simulation["gnss_rejected"]
    )

    print(
        "IMU dropouts:",
        simulation["imu_dropouts"]
    )

    print(
        "Barometer dropouts:",
        simulation["baro_dropouts"]
    )

    print(
        "Landing status:",
        status
    )


# ================================================================
# RESULTS TABLE
# ================================================================

print()
print("=" * 64)
print("V10.4 MULTI-SENSOR FAILURE RESULTS")
print("=" * 64)

print(
    "Scenario                         "
    "Landing Error   EKF Pos   "
    "EKF Alt   EKF Heading   Status"
)

print("-" * 90)

for result in results:

    status = (
        "PASS"
        if result[
            "true_landing_error"
        ] <= TARGET_TOLERANCE
        else
        "FAIL"
    )

    print(
        f"{result['scenario']:<32}"
        f"{result['true_landing_error']:>8.3f}       "
        f"{result['rms_position']:>7.3f}    "
        f"{result['rms_altitude']:>7.3f}    "
        f"{result['rms_heading']:>7.3f}       "
        f"{status}"
    )


# ================================================================
# BEST LANDING
# ================================================================

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


print()
print("=" * 64)
print("BEST LANDING CASE")
print("=" * 64)

print(
    "Failure mode:",
    best_landing["scenario"]
)

print(
    "Landing X:",
    f"{best_landing['landing_x']:.3f}",
    "m"
)

print(
    "Landing Y:",
    f"{best_landing['landing_y']:.3f}",
    "m"
)

print(
    "Landing error:",
    f"{best_landing['true_landing_error']:.3f}",
    "m"
)


# ================================================================
# WORST LANDING
# ================================================================

print()
print("=" * 64)
print("WORST LANDING CASE")
print("=" * 64)

print(
    "Failure mode:",
    worst_landing["scenario"]
)

print(
    "Landing X:",
    f"{worst_landing['landing_x']:.3f}",
    "m"
)

print(
    "Landing Y:",
    f"{worst_landing['landing_y']:.3f}",
    "m"
)

print(
    "Landing error:",
    f"{worst_landing['true_landing_error']:.3f}",
    "m"
)


# ================================================================
# BEST EKF
# ================================================================

best_ekf = min(
    results,
    key=lambda r:
    r["rms_position"]
)

worst_ekf = max(
    results,
    key=lambda r:
    r["rms_position"]
)


print()
print("=" * 64)
print("BEST EKF ESTIMATION CASE")
print("=" * 64)

print(
    "Failure mode:",
    best_ekf["scenario"]
)

print(
    "RMS position error:",
    f"{best_ekf['rms_position']:.3f}",
    "m"
)

print(
    "RMS altitude error:",
    f"{best_ekf['rms_altitude']:.3f}",
    "m"
)

print(
    "RMS heading error:",
    f"{best_ekf['rms_heading']:.3f}",
    "deg"
)


print()
print("=" * 64)
print("WORST EKF ESTIMATION CASE")
print("=" * 64)

print(
    "Failure mode:",
    worst_ekf["scenario"]
)

print(
    "RMS position error:",
    f"{worst_ekf['rms_position']:.3f}",
    "m"
)

print(
    "Maximum position error:",
    f"{worst_ekf['max_position']:.3f}",
    "m"
)


# ================================================================
# SUCCESS ANALYSIS
# ================================================================

successful_cases = [

    r for r in results
    if r["true_landing_error"]
    <= TARGET_TOLERANCE
]


print()
print("=" * 64)
print("MULTI-SENSOR ROBUSTNESS ASSESSMENT")
print("=" * 64)

print(
    "Successful scenarios:",
    len(successful_cases),
    "/",
    len(results)
)

print(
    "Success percentage:",
    f"{100.0 * len(successful_cases) / len(results):.1f}",
    "%"
)

if len(successful_cases) > 0:

    print()
    print(
        "Successful failure modes:"
    )

    for r in successful_cases:

        print(
            " -",
            r["scenario"],
            ":",
            f"{r['true_landing_error']:.3f}",
            "m"
        )


# ================================================================
# SENSOR FAILURE DIAGNOSTICS
# ================================================================

print()
print("=" * 64)
print("SENSOR FAILURE DIAGNOSTICS")
print("=" * 64)

for r in results:

    print()
    print(
        r["scenario"]
    )

    print(
        "GNSS dropouts:",
        r["gnss_dropouts"]
    )

    print(
        "GNSS rejected:",
        r["gnss_rejected"]
    )

    print(
        "IMU dropouts:",
        r["imu_dropouts"]
    )

    print(
        "Barometer dropouts:",
        r["baro_dropouts"]
    )

    print(
        "Maximum position error:",
        f"{r['max_position']:.3f}",
        "m"
    )


# ================================================================
# NAVIGATION SYSTEM ASSESSMENT
# ================================================================

print()
print("=" * 64)
print("V10.4 NAVIGATION SYSTEM ASSESSMENT")
print("=" * 64)

print(
    "Raw GNSS position noise:",
    GNSS_POSITION_NOISE,
    "m"
)

print(
    "Best EKF RMS position error:",
    f"{best_ekf['rms_position']:.3f}",
    "m"
)

if (
    best_ekf["rms_position"]
    <
    GNSS_POSITION_NOISE
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

print(
    "IMU FAILURE HANDLING: ACTIVE"
)

print(
    "BAROMETER FAILURE HANDLING: ACTIVE"
)


# ================================================================
# FINAL DIAGNOSTIC
# ================================================================

print()
print("=" * 64)
print("V10.4 DIAGNOSTIC SUMMARY")
print("=" * 64)

print()

print(
    "The simulation evaluates the ability of the navigation"
)

print(
    "system to maintain a usable state estimate when multiple"
)

print(
    "navigation sensors experience temporary failures."
)

print()

print(
    "GNSS provides absolute horizontal position and velocity."
)

print(
    "The IMU provides heading and turn-rate information."
)

print(
    "The barometer provides altitude information."
)

print()

print(
    "During sensor outages, the EKF propagates the state"
)

print(
    "using the available process model and remaining sensors."
)

print()

print(
    "When measurements return, the EKF corrects accumulated"
)

print(
    "state-estimation drift."
)

print()

if len(successful_cases) > 0:

    print(
        "At least one multi-sensor failure scenario remained"
    )

    print(
        "inside the specified landing tolerance."
    )

else:

    print(
        "No tested multi-sensor failure scenario remained"
    )

    print(
        "inside the specified landing tolerance."
    )

print()
print("=" * 64)
print("V10.4 SIMULATION COMPLETE")
print("=" * 64)

print(
    "Reference wind:",
    REFERENCE_WIND_SPEED,
    "m/s"
)

print(
    "Reference direction:",
    REFERENCE_WIND_DIRECTION,
    "degrees"
)

print(
    "Scenarios tested:",
    len(SCENARIOS)
)

print(
    "Best landing error:",
    f"{best_landing['true_landing_error']:.3f}",
    "m"
)

print(
    "Worst landing error:",
    f"{worst_landing['true_landing_error']:.3f}",
    "m"
)

print(
    "Best EKF RMS position:",
    f"{best_ekf['rms_position']:.3f}",
    "m"
)

print(
    "Worst EKF RMS position:",
    f"{worst_ekf['rms_position']:.3f}",
    "m"
)

print("=" * 64)

print()
print(
    "NEXT DEVELOPMENT STEP:"
)

print(
    "V10.5 -> Sensor fault detection + automatic sensor"
)

print(
    "         health monitoring + adaptive EKF measurement"
)

print(
    "         weighting"
)

print("=" * 64)