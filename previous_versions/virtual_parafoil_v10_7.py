"""
================================================================
VIRTUAL PARAFOIL V10.7
WIND-AWARE GUIDANCE + WIND UNCERTAINTY + ADAPTIVE PREDICTION
================================================================

Development from V10.6:
    1. Online wind estimation
    2. Wind-vector confidence / uncertainty estimation
    3. Adaptive wind filtering
    4. Wind-aware trajectory prediction
    5. Wind-aware candidate steering selection
    6. Adaptive prediction horizon
    7. Monte-Carlo wind robustness study
    8. No nested f-string syntax errors
================================================================
"""

import math
import random
import statistics


# ================================================================
# GLOBAL CONFIGURATION
# ================================================================

VERSION = "V10.7"

# ---------------- Parafoil ----------------
AREA = 0.96
MASS = 1.0
CL = 0.40
CD = 0.25

RHO = 1.225
G = 9.81

# Derived flight parameters
AIRSPEED = math.sqrt(
    (2.0 * MASS * G) /
    (RHO * AREA * math.sqrt(CL ** 2 + CD ** 2))
)

GLIDE_RATIO = CL / CD

HORIZONTAL_AIR_SPEED = AIRSPEED * (
    CL / math.sqrt(CL ** 2 + CD ** 2)
)

VERTICAL_DESCENT_SPEED = 3.717


# ---------------- Target ----------------
TARGET_X = 500.0
TARGET_Y = 200.0
LANDING_TOLERANCE = 20.0


# ---------------- Reference wind ----------------
REFERENCE_WIND_SPEED = 3.0
REFERENCE_WIND_DIRECTION_DEG = 0.0


# ---------------- Sensors ----------------
GNSS_POS_NOISE = 3.0
GNSS_VEL_NOISE = 0.30
BARO_NOISE = 2.0

IMU_HEADING_NOISE_DEG = 2.0
IMU_TURN_NOISE_DEG_S = 0.5

AIRSPEED_NOISE = 0.20


# ---------------- Update rates ----------------
GNSS_RATE = 5.0
BARO_RATE = 10.0
IMU_RATE = 50.0
AIRSPEED_RATE = 20.0


# ---------------- Simulation ----------------
DT = 1.0 / IMU_RATE
INITIAL_ALTITUDE = 600.0

MAX_TIME = 250.0


# ---------------- Wind estimator ----------------
WIND_ALPHA_MIN = 0.03
WIND_ALPHA_MAX = 0.25

MAX_WIND_ESTIMATE = 12.0

WIND_HEALTH_THRESHOLD = 0.20

INITIAL_WIND_EST_X = 0.0
INITIAL_WIND_EST_Y = 0.0


# ---------------- Guidance ----------------
GUIDANCE_INTERVAL = 2.0

NUM_CANDIDATE_COMMANDS = 21

MAX_TURN_RATE_DEG_S = 15.0
MAX_TURN_RATE = math.radians(MAX_TURN_RATE_DEG_S)

COMMAND_RANGE = MAX_TURN_RATE

# Candidate steering commands
CANDIDATE_COMMANDS = [
    -COMMAND_RANGE +
    i * (2.0 * COMMAND_RANGE / (NUM_CANDIDATE_COMMANDS - 1))
    for i in range(NUM_CANDIDATE_COMMANDS)
]


# ---------------- Prediction ----------------
HORIZON_HIGH = 20.0
HORIZON_MID = 15.0
HORIZON_LOW = 10.0
HORIZON_FINAL = 5.0


# ---------------- Monte Carlo ----------------
MONTE_CARLO_RUNS = 100

MC_MIN_WIND = 0.0
MC_MAX_WIND = 7.0

MC_SEED = 107


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def deg(rad):
    return math.degrees(rad)


def rad(degrees):
    return math.radians(degrees)


def wind_components(speed, direction_deg):
    direction = math.radians(direction_deg)

    return (
        speed * math.cos(direction),
        speed * math.sin(direction)
    )


def wind_speed_direction(wx, wy):

    speed = math.hypot(wx, wy)

    direction = math.degrees(
        math.atan2(wy, wx)
    )

    direction %= 360.0

    return speed, direction


# ================================================================
# TRUE VEHICLE MODEL
# ================================================================

class Parafoil:

    def __init__(self):

        self.x = 0.0
        self.y = 0.0

        self.altitude = INITIAL_ALTITUDE

        self.heading = 0.0

        self.turn_rate = 0.0

        self.time = 0.0

        self.total_steering = 0.0
        self.steering_commands = []

    # ------------------------------------------------------------

    def ground_velocity(self, wind_x, wind_y):

        air_vx = HORIZONTAL_AIR_SPEED * math.cos(self.heading)
        air_vy = HORIZONTAL_AIR_SPEED * math.sin(self.heading)

        return (
            air_vx + wind_x,
            air_vy + wind_y
        )

    # ------------------------------------------------------------

    def update(self, command, wind_x, wind_y):

        self.turn_rate = clamp(
            command,
            -MAX_TURN_RATE,
            MAX_TURN_RATE
        )

        self.heading = wrap_angle(
            self.heading +
            self.turn_rate * DT
        )

        vx, vy = self.ground_velocity(
            wind_x,
            wind_y
        )

        self.x += vx * DT
        self.y += vy * DT

        self.altitude -= VERTICAL_DESCENT_SPEED * DT

        self.time += DT

        self.total_steering += abs(
            self.turn_rate
        )

        self.steering_commands.append(
            self.turn_rate
        )


# ================================================================
# WIND ESTIMATOR
# ================================================================

class WindEstimator:

    def __init__(self):

        self.wx = INITIAL_WIND_EST_X
        self.wy = INITIAL_WIND_EST_Y

        self.variance_x = 4.0
        self.variance_y = 4.0

        self.health = 1.0

        self.error_history = []

    # ------------------------------------------------------------

    def adaptive_alpha(self):

        uncertainty = math.sqrt(
            max(
                self.variance_x +
                self.variance_y,
                0.0
            )
        )

        # Higher uncertainty -> faster response
        alpha = (
            WIND_ALPHA_MIN +
            0.08 * uncertainty
        )

        return clamp(
            alpha,
            WIND_ALPHA_MIN,
            WIND_ALPHA_MAX
        )

    # ------------------------------------------------------------

    def update(
        self,
        ground_vx,
        ground_vy,
        heading,
        airspeed,
        measurement_weight=1.0
    ):

        air_vx = (
            airspeed *
            math.cos(heading)
        )

        air_vy = (
            airspeed *
            math.sin(heading)
        )

        measured_wx = ground_vx - air_vx
        measured_wy = ground_vy - air_vy

        magnitude = math.hypot(
            measured_wx,
            measured_wy
        )

        if magnitude > MAX_WIND_ESTIMATE:

            self.health *= 0.95

            measurement_weight *= 0.1

        else:

            self.health += (
                0.01 *
                (1.0 - self.health)
            )

        measurement_weight = clamp(
            measurement_weight,
            0.0,
            1.0
        )

        alpha = self.adaptive_alpha()

        alpha *= measurement_weight

        self.wx += alpha * (
            measured_wx - self.wx
        )

        self.wy += alpha * (
            measured_wy - self.wy
        )

        residual_x = (
            measured_wx -
            self.wx
        )

        residual_y = (
            measured_wy -
            self.wy
        )

        self.variance_x = (
            0.98 * self.variance_x +
            0.02 * residual_x ** 2
        )

        self.variance_y = (
            0.98 * self.variance_y +
            0.02 * residual_y ** 2
        )

        self.wx = clamp(
            self.wx,
            -MAX_WIND_ESTIMATE,
            MAX_WIND_ESTIMATE
        )

        self.wy = clamp(
            self.wy,
            -MAX_WIND_ESTIMATE,
            MAX_WIND_ESTIMATE
        )

    # ------------------------------------------------------------

    def uncertainty(self):

        return math.sqrt(
            max(
                self.variance_x +
                self.variance_y,
                0.0
            )
        )

    # ------------------------------------------------------------

    def speed_direction(self):

        return wind_speed_direction(
            self.wx,
            self.wy
        )


# ================================================================
# SIMPLE EKF-LIKE NAVIGATION STATE
# ================================================================

class NavigationState:

    def __init__(self):

        self.x = 0.0
        self.y = 0.0

        self.vx = HORIZONTAL_AIR_SPEED
        self.vy = 0.0

        self.altitude = INITIAL_ALTITUDE

        self.heading = 0.0

        self.position_error_history = []
        self.altitude_error_history = []
        self.heading_error_history = []

    # ------------------------------------------------------------

    def predict(
        self,
        turn_rate,
        wind_x,
        wind_y
    ):

        self.heading = wrap_angle(
            self.heading +
            turn_rate * DT
        )

        air_vx = (
            HORIZONTAL_AIR_SPEED *
            math.cos(self.heading)
        )

        air_vy = (
            HORIZONTAL_AIR_SPEED *
            math.sin(self.heading)
        )

        self.vx = air_vx + wind_x
        self.vy = air_vy + wind_y

        self.x += self.vx * DT
        self.y += self.vy * DT

        self.altitude -= (
            VERTICAL_DESCENT_SPEED *
            DT
        )


# ================================================================
# SENSOR HEALTH
# ================================================================

class SensorHealth:

    def __init__(self):

        self.gnss = 1.0
        self.imu = 1.0
        self.baro = 1.0
        self.airspeed = 1.0

    # ------------------------------------------------------------

    def update(
        self,
        innovation,
        threshold,
        sensor
    ):

        normalized = (
            abs(innovation) /
            max(threshold, 1e-6)
        )

        if normalized > 1.0:

            decrease = clamp(
                0.03 * normalized,
                0.01,
                0.20
            )

            value = getattr(
                self,
                sensor
            )

            value -= decrease

        else:

            value = getattr(
                self,
                sensor
            )

            value += 0.01

        value = clamp(
            value,
            0.0,
            1.0
        )

        setattr(
            self,
            sensor,
            value
        )


# ================================================================
# SENSOR GENERATION
# ================================================================

def generate_gnss(
    true_x,
    true_y,
    true_vx,
    true_vy
):

    measured_x = (
        true_x +
        random.gauss(
            0.0,
            GNSS_POS_NOISE
        )
    )

    measured_y = (
        true_y +
        random.gauss(
            0.0,
            GNSS_POS_NOISE
        )
    )

    measured_vx = (
        true_vx +
        random.gauss(
            0.0,
            GNSS_VEL_NOISE
        )
    )

    measured_vy = (
        true_vy +
        random.gauss(
            0.0,
            GNSS_VEL_NOISE
        )
    )

    return (
        measured_x,
        measured_y,
        measured_vx,
        measured_vy
    )


def generate_baro(true_altitude):

    return (
        true_altitude +
        random.gauss(
            0.0,
            BARO_NOISE
        )
    )


def generate_heading(true_heading):

    return wrap_angle(
        true_heading +
        random.gauss(
            0.0,
            rad(IMU_HEADING_NOISE_DEG)
        )
    )


def generate_airspeed():

    return max(
        0.1,
        AIRSPEED +
        random.gauss(
            0.0,
            AIRSPEED_NOISE
        )
    )


# ================================================================
# WIND-AWARE PREDICTION
# ================================================================

def prediction_horizon(altitude):

    if altitude > 400.0:

        return HORIZON_HIGH

    if altitude > 200.0:

        return HORIZON_MID

    if altitude > 100.0:

        return HORIZON_LOW

    return HORIZON_FINAL


# ------------------------------------------------------------

def predict_landing_point(
    x,
    y,
    altitude,
    heading,
    steering,
    wind_x,
    wind_y
):

    remaining_time = max(
        altitude /
        VERTICAL_DESCENT_SPEED,
        0.0
    )

    horizon = prediction_horizon(
        altitude
    )

    prediction_time = min(
        remaining_time,
        horizon
    )

    predicted_heading = wrap_angle(
        heading +
        steering *
        prediction_time
    )

    air_vx = (
        HORIZONTAL_AIR_SPEED *
        math.cos(predicted_heading)
    )

    air_vy = (
        HORIZONTAL_AIR_SPEED *
        math.sin(predicted_heading)
    )

    predicted_vx = (
        air_vx +
        wind_x
    )

    predicted_vy = (
        air_vy +
        wind_y
    )

    predicted_x = (
        x +
        predicted_vx *
        prediction_time
    )

    predicted_y = (
        y +
        predicted_vy *
        prediction_time
    )

    return (
        predicted_x,
        predicted_y
    )


# ================================================================
# WIND UNCERTAINTY PENALTY
# ================================================================

def wind_uncertainty_penalty(
    predicted_x,
    predicted_y,
    target_x,
    target_y,
    wind_uncertainty,
    prediction_time
):

    position_uncertainty = (
        wind_uncertainty *
        prediction_time
    )

    target_error = distance(
        predicted_x,
        predicted_y,
        target_x,
        target_y
    )

    # Normalize uncertainty against
    # target miss distance.
    penalty = (
        0.15 *
        position_uncertainty
    )

    return (
        target_error +
        penalty
    )


# ================================================================
# WIND-AWARE GUIDANCE
# ================================================================

def select_steering_command(
    nav,
    wind_estimator,
    altitude
):

    best_command = 0.0
    best_cost = float("inf")

    horizon = prediction_horizon(
        altitude
    )

    wind_uncertainty = (
        wind_estimator.uncertainty()
    )

    for command in CANDIDATE_COMMANDS:

        px, py = predict_landing_point(
            nav.x,
            nav.y,
            altitude,
            nav.heading,
            command,
            wind_estimator.wx,
            wind_estimator.wy
        )

        target_cost = (
            wind_uncertainty_penalty(
                px,
                py,
                TARGET_X,
                TARGET_Y,
                wind_uncertainty,
                horizon
            )
        )

        heading_to_target = math.atan2(
            TARGET_Y - nav.y,
            TARGET_X - nav.x
        )

        heading_error = abs(
            wrap_angle(
                heading_to_target -
                (
                    nav.heading +
                    command *
                    horizon
                )
            )
        )

        control_penalty = (
            0.15 *
            abs(command) /
            MAX_TURN_RATE
        )

        total_cost = (
            target_cost +
            4.0 *
            heading_error +
            control_penalty
        )

        if total_cost < best_cost:

            best_cost = total_cost
            best_command = command

    return best_command


# ================================================================
# SINGLE SIMULATION
# ================================================================

def run_simulation(
    wind_speed,
    wind_direction_deg,
    seed=1
):

    random.seed(seed)

    wind_x, wind_y = wind_components(
        wind_speed,
        wind_direction_deg
    )

    vehicle = Parafoil()

    nav = NavigationState()

    wind_estimator = WindEstimator()

    health = SensorHealth()

    true_wind_errors = []

    steering_timer = 0.0

    current_command = 0.0

    gnss_timer = 0.0
    baro_timer = 0.0
    airspeed_timer = 0.0

    while (
        vehicle.altitude > 0.0
        and vehicle.time < MAX_TIME
    ):

        # --------------------------------------------------------
        # TRUE VEHICLE
        # --------------------------------------------------------

        vehicle.update(
            current_command,
            wind_x,
            wind_y
        )

        # --------------------------------------------------------
        # TRUE VELOCITY
        # --------------------------------------------------------

        true_vx, true_vy = (
            vehicle.ground_velocity(
                wind_x,
                wind_y
            )
        )

        # --------------------------------------------------------
        # IMU
        # --------------------------------------------------------

        measured_heading = (
            generate_heading(
                vehicle.heading
            )
        )

        heading_innovation = wrap_angle(
            measured_heading -
            nav.heading
        )

        health.update(
            deg(heading_innovation),
            8.0,
            "imu"
        )

        nav.heading = (
            measured_heading
            if health.imu > WIND_HEALTH_THRESHOLD
            else nav.heading
        )

        # --------------------------------------------------------
        # NAVIGATION PREDICTION
        # --------------------------------------------------------

        nav.predict(
            vehicle.turn_rate,
            wind_estimator.wx,
            wind_estimator.wy
        )

        # --------------------------------------------------------
        # GNSS
        # --------------------------------------------------------

        gnss_timer += DT

        if gnss_timer >= 1.0 / GNSS_RATE:

            gnss_timer = 0.0

            gx, gy, gvx, gvy = (
                generate_gnss(
                    vehicle.x,
                    vehicle.y,
                    true_vx,
                    true_vy
                )
            )

            position_innovation = distance(
                gx,
                gy,
                nav.x,
                nav.y
            )

            health.update(
                position_innovation,
                5.0,
                "gnss"
            )

            if health.gnss > WIND_HEALTH_THRESHOLD:

                nav.x += (
                    0.35 *
                    health.gnss *
                    (gx - nav.x)
                )

                nav.y += (
                    0.35 *
                    health.gnss *
                    (gy - nav.y)
                )

                nav.vx = (
                    0.50 * nav.vx +
                    0.50 * gvx
                )

                nav.vy = (
                    0.50 * nav.vy +
                    0.50 * gvy
                )

                # ------------------------------------------------
                # WIND ESTIMATION
                # ------------------------------------------------

                airspeed_measurement = (
                    generate_airspeed()
                )

                wind_estimator.update(
                    gvx,
                    gvy,
                    measured_heading,
                    airspeed_measurement,
                    health.gnss
                )

        # --------------------------------------------------------
        # BAROMETER
        # --------------------------------------------------------

        baro_timer += DT

        if baro_timer >= 1.0 / BARO_RATE:

            baro_timer = 0.0

            measured_altitude = (
                generate_baro(
                    vehicle.altitude
                )
            )

            altitude_innovation = (
                measured_altitude -
                nav.altitude
            )

            health.update(
                altitude_innovation,
                5.0,
                "baro"
            )

            if health.baro > WIND_HEALTH_THRESHOLD:

                nav.altitude += (
                    0.35 *
                    health.baro *
                    altitude_innovation
                )

        # --------------------------------------------------------
        # AIRSPEED UPDATE
        # --------------------------------------------------------

        airspeed_timer += DT

        if airspeed_timer >= 1.0 / AIRSPEED_RATE:

            airspeed_timer = 0.0

            measured_airspeed = (
                generate_airspeed()
            )

            airspeed_innovation = (
                measured_airspeed -
                AIRSPEED
            )

            health.update(
                airspeed_innovation,
                1.0,
                "airspeed"
            )

        # --------------------------------------------------------
        # WIND ERROR
        # --------------------------------------------------------

        wind_error = math.hypot(
            wind_estimator.wx -
            wind_x,

            wind_estimator.wy -
            wind_y
        )

        true_wind_errors.append(
            wind_error
        )

        # --------------------------------------------------------
        # GUIDANCE
        # --------------------------------------------------------

        steering_timer += DT

        if steering_timer >= GUIDANCE_INTERVAL:

            steering_timer = 0.0

            current_command = (
                select_steering_command(
                    nav,
                    wind_estimator,
                    vehicle.altitude
                )
            )

    # ============================================================
    # FINAL RESULTS
    # ============================================================

    true_landing_error = distance(
        vehicle.x,
        vehicle.y,
        TARGET_X,
        TARGET_Y
    )

    estimated_landing_error = distance(
        nav.x,
        nav.y,
        TARGET_X,
        TARGET_Y
    )

    mean_wind_error = (
        statistics.mean(
            true_wind_errors
        )
        if true_wind_errors
        else 0.0
    )

    rms_wind_error = math.sqrt(
        statistics.mean(
            [
                e * e
                for e in true_wind_errors
            ]
        )
        if true_wind_errors
        else 0.0
    )

    return {
        "wind_speed": wind_speed,
        "wind_direction": wind_direction_deg,

        "wind_x": wind_x,
        "wind_y": wind_y,

        "landing_x": vehicle.x,
        "landing_y": vehicle.y,

        "estimated_x": nav.x,
        "estimated_y": nav.y,

        "true_landing_error":
            true_landing_error,

        "estimated_landing_error":
            estimated_landing_error,

        "flight_time":
            vehicle.time,

        "estimated_wind_x":
            wind_estimator.wx,

        "estimated_wind_y":
            wind_estimator.wy,

        "estimated_wind_speed":
            wind_estimator.speed_direction()[0],

        "estimated_wind_direction":
            wind_estimator.speed_direction()[1],

        "mean_wind_error":
            mean_wind_error,

        "wind_rms_error":
            rms_wind_error,

        "gnss_health":
            health.gnss,

        "imu_health":
            health.imu,

        "baro_health":
            health.baro,

        "airspeed_health":
            health.airspeed,

        "ekf_rms_position":
            0.0,

        "ekf_rms_altitude":
            0.0,

        "ekf_rms_heading":
            0.0,

        "steering_commands":
            vehicle.steering_commands
    }


# ================================================================
# PRINT REFERENCE RESULTS
# ================================================================

def print_reference_result(result):

    print()
    print("=" * 72)
    print("V10.7 REFERENCE RESULTS")
    print("=" * 72)

    print(
        f"True landing X: "
        f"{result['landing_x']:.3f} m"
    )

    print(
        f"True landing Y: "
        f"{result['landing_y']:.3f} m"
    )

    print(
        f"Estimated landing X: "
        f"{result['estimated_x']:.3f} m"
    )

    print(
        f"Estimated landing Y: "
        f"{result['estimated_y']:.3f} m"
    )

    print(
        f"Target: "
        f"{TARGET_X:.3f}, "
        f"{TARGET_Y:.3f} m"
    )

    print(
        f"True landing error: "
        f"{result['true_landing_error']:.3f} m"
    )

    print(
        f"Estimated landing error: "
        f"{result['estimated_landing_error']:.3f} m"
    )

    print(
        f"Flight time: "
        f"{result['flight_time']:.3f} s"
    )

    print()
    print("--- WIND ESTIMATION ---")

    print(
        f"True wind X: "
        f"{result['wind_x']:.3f} m/s"
    )

    print(
        f"True wind Y: "
        f"{result['wind_y']:.3f} m/s"
    )

    print(
        f"True wind speed: "
        f"{result['wind_speed']:.3f} m/s"
    )

    print(
        f"True wind direction: "
        f"{result['wind_direction']:.3f} deg"
    )

    print(
        f"Estimated wind X: "
        f"{result['estimated_wind_x']:.3f} m/s"
    )

    print(
        f"Estimated wind Y: "
        f"{result['estimated_wind_y']:.3f} m/s"
    )

    print(
        f"Estimated wind speed: "
        f"{result['estimated_wind_speed']:.3f} m/s"
    )

    print(
        f"Estimated wind direction: "
        f"{result['estimated_wind_direction']:.3f} deg"
    )

    print(
        f"Mean wind estimation error: "
        f"{result['mean_wind_error']:.3f} m/s"
    )

    print(
        f"RMS wind estimation error: "
        f"{result['wind_rms_error']:.3f} m/s"
    )

    print()
    print("--- SENSOR HEALTH ---")

    print(
        f"Final GNSS health: "
        f"{result['gnss_health']:.3f}"
    )

    print(
        f"Final IMU health: "
        f"{result['imu_health']:.3f}"
    )

    print(
        f"Final barometer health: "
        f"{result['baro_health']:.3f}"
    )

    print(
        f"Final airspeed health: "
        f"{result['airspeed_health']:.3f}"
    )

    status = (
        "WITHIN TOLERANCE"
        if result["true_landing_error"]
        <= LANDING_TOLERANCE
        else "OUTSIDE TOLERANCE"
    )

    print()
    print(
        f"Landing status: {status}"
    )


# ================================================================
# WIND DIRECTION STUDY
# ================================================================

def wind_direction_study():

    print()
    print("=" * 72)
    print("V10.7 WIND DIRECTION STUDY")
    print("=" * 72)

    print(
        "Wind speed fixed at "
        f"{REFERENCE_WIND_SPEED:.1f} m/s"
    )

    directions = [
        0.0,
        45.0,
        90.0,
        135.0,
        180.0,
        225.0,
        270.0,
        315.0
    ]

    results = []

    for i, direction in enumerate(
        directions,
        start=1
    ):

        result = run_simulation(
            REFERENCE_WIND_SPEED,
            direction,
            seed=1000 + i
        )

        results.append(result)

        print(
            f"Simulation {i}/8 | "
            f"Direction = {direction:6.1f} deg | "
            f"Landing Error = "
            f"{result['true_landing_error']:8.3f} m | "
            f"Wind RMS Error = "
            f"{result['wind_rms_error']:7.3f} m/s"
        )

    return results


# ================================================================
# WIND SPEED STUDY
# ================================================================

def wind_speed_study():

    print()
    print("=" * 72)
    print("V10.7 WIND SPEED STUDY")
    print("=" * 72)

    print(
        "Wind direction fixed at "
        f"{REFERENCE_WIND_DIRECTION_DEG:.1f} degrees"
    )

    speeds = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0
    ]

    results = []

    for i, speed in enumerate(
        speeds,
        start=1
    ):

        result = run_simulation(
            speed,
            REFERENCE_WIND_DIRECTION_DEG,
            seed=2000 + i
        )

        results.append(result)

        print(
            f"Simulation {i}/8 | "
            f"Wind = {speed:5.2f} m/s | "
            f"Landing Error = "
            f"{result['true_landing_error']:8.3f} m | "
            f"Estimated Wind = "
            f"{result['estimated_wind_speed']:6.3f} m/s | "
            f"Wind RMS = "
            f"{result['wind_rms_error']:7.3f} m/s"
        )

    return results


# ================================================================
# MONTE CARLO VALIDATION
# ================================================================

def monte_carlo_validation():

    print()
    print("=" * 72)
    print("V10.7 MONTE CARLO WIND VALIDATION")
    print("=" * 72)

    print(
        f"Number of simulations: "
        f"{MONTE_CARLO_RUNS}"
    )

    print(
        f"Wind speed range: "
        f"{MC_MIN_WIND:.1f} - "
        f"{MC_MAX_WIND:.1f} m/s"
    )

    print(
        "Wind direction range: 0 - 360 degrees"
    )

    random.seed(MC_SEED)

    results = []

    for i in range(
        1,
        MONTE_CARLO_RUNS + 1
    ):

        speed = random.uniform(
            MC_MIN_WIND,
            MC_MAX_WIND
        )

        direction = random.uniform(
            0.0,
            360.0
        )

        result = run_simulation(
            speed,
            direction,
            seed=MC_SEED + i
        )

        results.append(result)

        if (
            i == 1
            or i % 10 == 0
            or i == MONTE_CARLO_RUNS
        ):

            print(
                f"Simulation {i:3d}/"
                f"{MONTE_CARLO_RUNS} | "
                f"Wind = {speed:5.2f} m/s | "
                f"Direction = {direction:6.1f} deg | "
                f"Landing Error = "
                f"{result['true_landing_error']:8.2f} m | "
                f"Wind RMS = "
                f"{result['wind_rms_error']:5.2f} m/s"
            )

    return results


# ================================================================
# MONTE CARLO RESULTS
# ================================================================

def print_monte_carlo_results(
    results
):

    landing_errors = [
        r["true_landing_error"]
        for r in results
    ]

    wind_errors = [
        r["wind_rms_error"]
        for r in results
    ]

    mean_landing = statistics.mean(
        landing_errors
    )

    median_landing = statistics.median(
        landing_errors
    )

    std_landing = (
        statistics.stdev(
            landing_errors
        )
        if len(landing_errors) > 1
        else 0.0
    )

    min_landing = min(
        landing_errors
    )

    max_landing = max(
        landing_errors
    )

    mean_wind = statistics.mean(
        wind_errors
    )

    median_wind = statistics.median(
        wind_errors
    )

    within_5 = sum(
        1
        for e in landing_errors
        if e <= 5.0
    )

    within_10 = sum(
        1
        for e in landing_errors
        if e <= 10.0
    )

    within_20 = sum(
        1
        for e in landing_errors
        if e <= LANDING_TOLERANCE
    )

    print()
    print("=" * 72)
    print("V10.7 MONTE CARLO RESULTS")
    print("=" * 72)

    print(
        f"Number of simulations: "
        f"{len(results)}"
    )

    print(
        f"Mean landing error: "
        f"{mean_landing:.3f} m"
    )

    print(
        f"Median landing error: "
        f"{median_landing:.3f} m"
    )

    print(
        f"Standard deviation: "
        f"{std_landing:.3f} m"
    )

    print(
        f"Minimum landing error: "
        f"{min_landing:.3f} m"
    )

    print(
        f"Maximum landing error: "
        f"{max_landing:.3f} m"
    )

    print(
        f"Landing within 5 m: "
        f"{100.0 * within_5 / len(results):.2f} %"
    )

    print(
        f"Landing within 10 m: "
        f"{100.0 * within_10 / len(results):.2f} %"
    )

    print(
        f"Landing within 20 m: "
        f"{100.0 * within_20 / len(results):.2f} %"
    )

    print(
        f"Mean wind RMS error: "
        f"{mean_wind:.3f} m/s"
    )

    print(
        f"Median wind RMS error: "
        f"{median_wind:.3f} m/s"
    )

    # ------------------------------------------------------------
    # BEST LANDING
    # ------------------------------------------------------------

    best = min(
        results,
        key=lambda r:
        r["true_landing_error"]
    )

    worst = max(
        results,
        key=lambda r:
        r["true_landing_error"]
    )

    best_wind = min(
        results,
        key=lambda r:
        r["wind_rms_error"]
    )

    worst_wind = max(
        results,
        key=lambda r:
        r["wind_rms_error"]
    )

    print()
    print("BEST LANDING CASE")

    print(
        f"Wind speed: "
        f"{best['wind_speed']:.3f} m/s"
    )

    print(
        f"Wind direction: "
        f"{best['wind_direction']:.3f} degrees"
    )

    print(
        f"Landing X: "
        f"{best['landing_x']:.3f} m"
    )

    print(
        f"Landing Y: "
        f"{best['landing_y']:.3f} m"
    )

    print(
        f"Landing error: "
        f"{best['true_landing_error']:.3f} m"
    )

    print()
    print("WORST LANDING CASE")

    print(
        f"Wind speed: "
        f"{worst['wind_speed']:.3f} m/s"
    )

    print(
        f"Wind direction: "
        f"{worst['wind_direction']:.3f} degrees"
    )

    print(
        f"Landing X: "
        f"{worst['landing_x']:.3f} m"
    )

    print(
        f"Landing Y: "
        f"{worst['landing_y']:.3f} m"
    )

    print(
        f"Landing error: "
        f"{worst['true_landing_error']:.3f} m"
    )

    print()
    print("BEST WIND ESTIMATION CASE")

    print(
        f"Wind speed: "
        f"{best_wind['wind_speed']:.3f} m/s"
    )

    print(
        f"Wind direction: "
        f"{best_wind['wind_direction']:.3f} degrees"
    )

    print(
        f"Wind RMS estimation error: "
        f"{best_wind['wind_rms_error']:.3f} m/s"
    )

    print()
    print("WORST WIND ESTIMATION CASE")

    print(
        f"Wind speed: "
        f"{worst_wind['wind_speed']:.3f} m/s"
    )

    print(
        f"Wind direction: "
        f"{worst_wind['wind_direction']:.3f} degrees"
    )

    print(
        f"Wind RMS estimation error: "
        f"{worst_wind['wind_rms_error']:.3f} m/s"
    )

    return {
        "mean_landing":
            mean_landing,

        "median_landing":
            median_landing,

        "std_landing":
            std_landing,

        "min_landing":
            min_landing,

        "max_landing":
            max_landing,

        "within_5":
            100.0 *
            within_5 /
            len(results),

        "within_10":
            100.0 *
            within_10 /
            len(results),

        "within_20":
            100.0 *
            within_20 /
            len(results),

        "mean_wind":
            mean_wind,

        "median_wind":
            median_wind,

        "best":
            best,

        "worst":
            worst,

        "best_wind":
            best_wind,

        "worst_wind":
            worst_wind
    }


# ================================================================
# FINAL ASSESSMENT
# ================================================================

def print_final_assessment(
    reference_result,
    monte_carlo_result
):

    print()
    print("=" * 72)
    print("V10.7 NAVIGATION SYSTEM ASSESSMENT")
    print("=" * 72)

    print(
        f"Raw GNSS position noise: "
        f"{GNSS_POS_NOISE:.1f} m"
    )

    print(
        f"Reference wind RMS estimation error: "
        f"{reference_result['wind_rms_error']:.3f} m/s"
    )

    print(
        f"Reference landing error: "
        f"{reference_result['true_landing_error']:.3f} m"
    )

    print(
        f"Monte Carlo mean wind RMS error: "
        f"{monte_carlo_result['mean_wind']:.3f} m/s"
    )

    print(
        f"Monte Carlo landing success <=20 m: "
        f"{monte_carlo_result['within_20']:.2f} %"
    )

    print(
        "ONLINE WIND ESTIMATION: ACTIVE"
    )

    print(
        "ADAPTIVE WIND VECTOR: ACTIVE"
    )

    print(
        "WIND UNCERTAINTY MODEL: ACTIVE"
    )

    print(
        "WIND-AWARE GUIDANCE: ACTIVE"
    )

    print(
        "ADAPTIVE PREDICTION HORIZON: ACTIVE"
    )

    print()
    print("V10.7 STATUS: COMPLETE")


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 72)
    print("VIRTUAL PARAFOIL V10.7")
    print("WIND-AWARE GUIDANCE + WIND UNCERTAINTY")
    print("=" * 72)

    print()
    print("PARAFOIL PARAMETERS")

    print(
        f"Area: {AREA} m^2"
    )

    print(
        f"Mass: {MASS} kg"
    )

    print(
        f"CL: {CL}"
    )

    print(
        f"CD: {CD}"
    )

    print(
        f"Airspeed: {AIRSPEED:.6f} m/s"
    )

    print(
        f"Horizontal air velocity: "
        f"{HORIZONTAL_AIR_SPEED:.6f} m/s"
    )

    print(
        f"Vertical descent velocity: "
        f"{VERTICAL_DESCENT_SPEED:.3f} m/s"
    )

    print(
        f"Glide ratio: "
        f"{GLIDE_RATIO:.3f}"
    )

    print()
    print("TARGET")

    print(
        f"Target X: {TARGET_X:.1f} m"
    )

    print(
        f"Target Y: {TARGET_Y:.1f} m"
    )

    print(
        f"Tolerance: "
        f"{LANDING_TOLERANCE:.1f} m"
    )

    print()
    print("REFERENCE WIND")

    print(
        f"Wind speed: "
        f"{REFERENCE_WIND_SPEED:.1f} m/s"
    )

    print(
        f"Wind direction: "
        f"{REFERENCE_WIND_DIRECTION_DEG:.1f} degrees"
    )

    print()
    print("SENSOR MODEL")

    print(
        f"GNSS position noise: "
        f"{GNSS_POS_NOISE:.1f} m"
    )

    print(
        f"GNSS velocity noise: "
        f"{GNSS_VEL_NOISE:.2f} m/s"
    )

    print(
        f"Barometer noise: "
        f"{BARO_NOISE:.1f} m"
    )

    print(
        f"IMU heading noise: "
        f"{IMU_HEADING_NOISE_DEG:.1f} deg"
    )

    print(
        f"IMU turn-rate noise: "
        f"{IMU_TURN_NOISE_DEG_S:.1f} deg/s"
    )

    print(
        f"Airspeed noise: "
        f"{AIRSPEED_NOISE:.2f} m/s"
    )

    print()
    print("UPDATE RATES")

    print(
        f"GNSS: {GNSS_RATE:.1f} Hz"
    )

    print(
        f"Barometer: {BARO_RATE:.1f} Hz"
    )

    print(
        f"IMU: {IMU_RATE:.1f} Hz"
    )

    print(
        f"Airspeed: {AIRSPEED_RATE:.1f} Hz"
    )

    print()
    print("V10.7 DEVELOPMENT")

    print(
        "Online wind estimation: ENABLED"
    )

    print(
        "Wind uncertainty estimation: ENABLED"
    )

    print(
        "Adaptive wind filtering: ENABLED"
    )

    print(
        "Wind-aware guidance: ENABLED"
    )

    print(
        "Adaptive prediction horizon: ENABLED"
    )

    print()
    print(
        "STARTING V10.7 REFERENCE "
        "WIND SIMULATION"
    )

    reference_result = run_simulation(
        REFERENCE_WIND_SPEED,
        REFERENCE_WIND_DIRECTION_DEG,
        seed=500
    )

    print_reference_result(
        reference_result
    )

    # ------------------------------------------------------------
    # Direction study
    # ------------------------------------------------------------

    wind_direction_study()

    # ------------------------------------------------------------
    # Speed study
    # ------------------------------------------------------------

    wind_speed_study()

    # ------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------

    monte_carlo_results = (
        monte_carlo_validation()
    )

    monte_carlo_summary = (
        print_monte_carlo_results(
            monte_carlo_results
        )
    )

    # ------------------------------------------------------------
    # Final assessment
    # ------------------------------------------------------------

    print_final_assessment(
        reference_result,
        monte_carlo_summary
    )

    print()
    print("=" * 72)
    print("V10.7 SIMULATION COMPLETE")
    print("=" * 72)

    print(
        f"Reference wind: "
        f"{REFERENCE_WIND_SPEED:.1f} m/s"
    )

    print(
        f"Reference direction: "
        f"{REFERENCE_WIND_DIRECTION_DEG:.1f} degrees"
    )

    print(
        "Wind estimator: ONLINE"
    )

    print(
        "Adaptive wind-vector estimation: ACTIVE"
    )

    print(
        "Wind-aware guidance: ACTIVE"
    )

    print(
        "Monte Carlo simulations: "
        f"{MONTE_CARLO_RUNS}"
    )

    print("=" * 72)


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()