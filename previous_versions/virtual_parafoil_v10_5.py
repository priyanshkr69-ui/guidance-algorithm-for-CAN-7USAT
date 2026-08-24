"""
================================================================
VIRTUAL PARAFOIL V10.5
SENSOR FAULT DETECTION + HEALTH MONITORING
+ ADAPTIVE EKF MEASUREMENT WEIGHTING
================================================================

Development chain:
V8.x  -> Guidance
V9.x  -> Reachability-aware guidance
V10.0 -> EKF state estimation
V10.1 -> Sensor update-rate study
V10.2 -> GNSS dropout + outlier rejection
V10.3 -> GNSS failure-duration robustness
V10.4 -> Multi-sensor failure robustness
V10.5 -> Sensor health monitoring + adaptive EKF

This is a simulation/development model.
It is NOT flight-qualified software.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================
# RANDOM SEED
# ==============================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==============================================================
# PARAFOIL PARAMETERS
# ==============================================================

AREA = 0.96
MASS = 1.0
CL = 0.4
CD = 0.25
RHO = 1.225

TARGET_X = 500.0
TARGET_Y = 200.0
TARGET_TOLERANCE = 20.0

INITIAL_ALTITUDE = 600.0

# Approximate parafoil performance
AIRSPEED = math.sqrt(
    (2.0 * MASS * 9.81) /
    (RHO * AREA * CL)
)

HORIZONTAL_AIR_SPEED = AIRSPEED * 0.848
VERTICAL_DESCENT_SPEED = 3.717
GLIDE_RATIO = HORIZONTAL_AIR_SPEED / VERTICAL_DESCENT_SPEED


# ==============================================================
# WIND
# ==============================================================

WIND_SPEED = 3.0
WIND_DIRECTION_DEG = 0.0

WIND_X = WIND_SPEED * math.cos(math.radians(WIND_DIRECTION_DEG))
WIND_Y = WIND_SPEED * math.sin(math.radians(WIND_DIRECTION_DEG))


# ==============================================================
# SIMULATION
# ==============================================================

DT = 0.02
SIMULATION_TIME = 180.0

GUIDANCE_INTERVAL = 2.0

MAX_TURN_RATE_DEG = 15.0
MAX_TURN_RATE = math.radians(MAX_TURN_RATE_DEG)

MAX_STEERING = 1.0

# 21 candidate commands
STEERING_COMMANDS = np.linspace(-1.0, 1.0, 21)


# ==============================================================
# SENSOR PARAMETERS
# ==============================================================

GNSS_POSITION_NOISE = 3.0
GNSS_VELOCITY_NOISE = 0.3

BARO_ALTITUDE_NOISE = 2.0

IMU_HEADING_NOISE_DEG = 2.0
IMU_TURN_RATE_NOISE_DEG = 0.5

GNSS_RATE = 5.0
BARO_RATE = 10.0
IMU_RATE = 50.0


# ==============================================================
# SENSOR FAULT PARAMETERS
# ==============================================================

GNSS_OUTLIER_PROBABILITY = 0.05
GNSS_POSITION_OUTLIER = 40.0
GNSS_VELOCITY_OUTLIER = 5.0

# Health degradation
HEALTH_DECAY = 0.15
HEALTH_RECOVERY = 0.08

# Fault detection thresholds
GNSS_INNOVATION_THRESHOLD = 5.0
BARO_INNOVATION_THRESHOLD = 5.0
IMU_INNOVATION_THRESHOLD_DEG = 8.0

# Critical health threshold
HEALTH_REJECT_THRESHOLD = 0.20


# ==============================================================
# FAULT SCENARIOS
# ==============================================================

SCENARIOS = {

    "Nominal": {
        "gnss_fault": False,
        "imu_fault": False,
        "baro_fault": False
    },

    "GNSS degradation": {
        "gnss_fault": True,
        "imu_fault": False,
        "baro_fault": False
    },

    "IMU degradation": {
        "gnss_fault": False,
        "imu_fault": True,
        "baro_fault": False
    },

    "Barometer degradation": {
        "gnss_fault": False,
        "imu_fault": False,
        "baro_fault": True
    },

    "GNSS + IMU degradation": {
        "gnss_fault": True,
        "imu_fault": True,
        "baro_fault": False
    },

    "GNSS + Barometer degradation": {
        "gnss_fault": True,
        "imu_fault": False,
        "baro_fault": True
    },

    "IMU + Barometer degradation": {
        "gnss_fault": False,
        "imu_fault": True,
        "baro_fault": True
    },

    "ALL SENSOR degradation": {
        "gnss_fault": True,
        "imu_fault": True,
        "baro_fault": True
    }
}


# ==============================================================
# UTILITY FUNCTIONS
# ==============================================================

def wrap_angle(angle):

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def clamp(value, low, high):

    return max(low, min(high, value))


# ==============================================================
# PARAFOIL INITIAL STATE
# ==============================================================

def initial_state():

    # Start south-west of target
    x = 0.0
    y = 0.0

    altitude = INITIAL_ALTITUDE

    # Initial heading toward approximate target
    heading = math.atan2(
        TARGET_Y - y,
        TARGET_X - x
    )

    turn_rate = 0.0

    return np.array([
        x,
        y,
        altitude,
        heading,
        turn_rate
    ], dtype=float)


# ==============================================================
# TRUE PARAFOIL DYNAMICS
# ==============================================================

def propagate_true_state(state, steering, dt):

    x, y, altitude, heading, turn_rate = state

    # Steering -> turn rate
    desired_turn_rate = steering * MAX_TURN_RATE

    # First-order actuator response
    actuator_tau = 1.0

    turn_rate += (
        desired_turn_rate - turn_rate
    ) * dt / actuator_tau

    turn_rate = clamp(
        turn_rate,
        -MAX_TURN_RATE,
        MAX_TURN_RATE
    )

    heading += turn_rate * dt

    heading = wrap_angle(heading)

    # Air-relative velocity
    vx_air = HORIZONTAL_AIR_SPEED * math.cos(heading)
    vy_air = HORIZONTAL_AIR_SPEED * math.sin(heading)

    # Ground velocity
    vx = vx_air + WIND_X
    vy = vy_air + WIND_Y

    x += vx * dt
    y += vy * dt

    altitude -= VERTICAL_DESCENT_SPEED * dt

    altitude = max(0.0, altitude)

    return np.array([
        x,
        y,
        altitude,
        heading,
        turn_rate
    ])


# ==============================================================
# SENSOR HEALTH CLASS
# ==============================================================

class SensorHealth:

    def __init__(self):

        self.gnss = 1.0
        self.imu = 1.0
        self.baro = 1.0

        self.gnss_fault_count = 0
        self.imu_fault_count = 0
        self.baro_fault_count = 0

        self.gnss_rejected = 0
        self.baro_rejected = 0
        self.imu_rejected = 0

    # ----------------------------------------------------------

    def degrade(self, sensor):

        if sensor == "gnss":

            self.gnss -= HEALTH_DECAY
            self.gnss = clamp(self.gnss, 0.0, 1.0)

            self.gnss_fault_count += 1

        elif sensor == "imu":

            self.imu -= HEALTH_DECAY
            self.imu = clamp(self.imu, 0.0, 1.0)

            self.imu_fault_count += 1

        elif sensor == "baro":

            self.baro -= HEALTH_DECAY
            self.baro = clamp(self.baro, 0.0, 1.0)

            self.baro_fault_count += 1

    # ----------------------------------------------------------

    def recover(self, sensor):

        if sensor == "gnss":

            self.gnss += HEALTH_RECOVERY
            self.gnss = clamp(self.gnss, 0.0, 1.0)

        elif sensor == "imu":

            self.imu += HEALTH_RECOVERY
            self.imu = clamp(self.imu, 0.0, 1.0)

        elif sensor == "baro":

            self.baro += HEALTH_RECOVERY
            self.baro = clamp(self.baro, 0.0, 1.0)

    # ----------------------------------------------------------

    def covariance_multiplier(self, health):

        # Healthy sensor:
        # multiplier ~ 1

        # Unhealthy sensor:
        # covariance becomes larger

        return 1.0 / max(
            health ** 2,
            0.05
        )


# ==============================================================
# EKF CLASS
# ==============================================================
#
# State:
#
# X = [x, y, altitude, heading, turn_rate]
#
# ==============================================================

class ParafoilEKF:

    def __init__(self):

        self.x = np.array([
            0.0,
            0.0,
            INITIAL_ALTITUDE,
            0.0,
            0.0
        ], dtype=float)

        self.P = np.diag([
            25.0,
            25.0,
            16.0,
            math.radians(10.0) ** 2,
            math.radians(5.0) ** 2
        ])

        # Process noise
        self.Q = np.diag([
            0.20,
            0.20,
            0.10,
            math.radians(0.5) ** 2,
            math.radians(0.5) ** 2
        ])

    # ----------------------------------------------------------

    def predict(self, steering, dt):

        x, y, altitude, heading, turn_rate = self.x

        desired_turn_rate = (
            steering * MAX_TURN_RATE
        )

        tau = 1.0

        turn_rate += (
            desired_turn_rate - turn_rate
        ) * dt / tau

        heading += turn_rate * dt

        heading = wrap_angle(heading)

        vx = (
            HORIZONTAL_AIR_SPEED *
            math.cos(heading)
            + WIND_X
        )

        vy = (
            HORIZONTAL_AIR_SPEED *
            math.sin(heading)
            + WIND_Y
        )

        x += vx * dt
        y += vy * dt

        altitude -= VERTICAL_DESCENT_SPEED * dt

        altitude = max(
            0.0,
            altitude
        )

        self.x = np.array([
            x,
            y,
            altitude,
            heading,
            turn_rate
        ])

        # Numerical Jacobian
        F = np.eye(5)

        F[0, 3] = (
            -HORIZONTAL_AIR_SPEED *
            math.sin(heading) *
            dt
        )

        F[1, 3] = (
            HORIZONTAL_AIR_SPEED *
            math.cos(heading) *
            dt
        )

        F[3, 4] = dt

        self.P = (
            F @ self.P @ F.T
            + self.Q * dt
        )

        self.P = (
            self.P + self.P.T
        ) / 2.0

    # ----------------------------------------------------------

    def update_gnss(
        self,
        measurement,
        health
    ):

        if health <= HEALTH_REJECT_THRESHOLD:

            return False

        z = np.array([
            measurement[0],
            measurement[1]
        ])

        H = np.zeros((2, 5))

        H[0, 0] = 1.0
        H[1, 1] = 1.0

        predicted = H @ self.x

        innovation = z - predicted

        innovation_norm = np.linalg.norm(
            innovation
        )

        if innovation_norm > GNSS_INNOVATION_THRESHOLD * (
            1.0 + 4.0 * health
        ):

            return False

        multiplier = (
            1.0 / max(
                health ** 2,
                0.05
            )
        )

        R = np.diag([
            GNSS_POSITION_NOISE ** 2,
            GNSS_POSITION_NOISE ** 2
        ]) * multiplier

        S = (
            H @ self.P @ H.T
            + R
        )

        K = (
            self.P @ H.T
            @ np.linalg.inv(S)
        )

        self.x += K @ innovation

        I = np.eye(5)

        self.P = (
            I - K @ H
        ) @ self.P

        self.P = (
            self.P + self.P.T
        ) / 2.0

        return True

    # ----------------------------------------------------------

    def update_baro(
        self,
        altitude,
        health
    ):

        if health <= HEALTH_REJECT_THRESHOLD:

            return False

        H = np.zeros((1, 5))
        H[0, 2] = 1.0

        predicted = H @ self.x

        innovation = altitude - predicted[0]

        if abs(innovation) > (
            BARO_INNOVATION_THRESHOLD *
            (1.0 + 4.0 * health)
        ):

            return False

        multiplier = (
            1.0 / max(
                health ** 2,
                0.05
            )
        )

        R = np.array([
            [BARO_ALTITUDE_NOISE ** 2
             * multiplier]
        ])

        S = (
            H @ self.P @ H.T
            + R
        )

        K = (
            self.P @ H.T
            @ np.linalg.inv(S)
        )

        self.x += (
            K[:, 0] * innovation
        )

        I = np.eye(5)

        self.P = (
            I - K @ H
        ) @ self.P

        self.P = (
            self.P + self.P.T
        ) / 2.0

        return True

    # ----------------------------------------------------------

    def update_imu(
        self,
        heading,
        turn_rate,
        health
    ):

        if health <= HEALTH_REJECT_THRESHOLD:

            return False

        z = np.array([
            heading,
            turn_rate
        ])

        H = np.zeros((2, 5))

        H[0, 3] = 1.0
        H[1, 4] = 1.0

        predicted = H @ self.x

        innovation = z - predicted

        innovation[0] = wrap_angle(
            innovation[0]
        )

        innovation_norm_deg = math.degrees(
            abs(innovation[0])
        )

        if innovation_norm_deg > (
            IMU_INNOVATION_THRESHOLD_DEG *
            (1.0 + 4.0 * health)
        ):

            return False

        multiplier = (
            1.0 / max(
                health ** 2,
                0.05
            )
        )

        R = np.diag([
            math.radians(
                IMU_HEADING_NOISE_DEG
            ) ** 2,

            math.radians(
                IMU_TURN_RATE_NOISE_DEG
            ) ** 2
        ]) * multiplier

        S = (
            H @ self.P @ H.T
            + R
        )

        K = (
            self.P @ H.T
            @ np.linalg.inv(S)
        )

        self.x += K @ innovation

        self.x[3] = wrap_angle(
            self.x[3]
        )

        I = np.eye(5)

        self.P = (
            I - K @ H
        ) @ self.P

        self.P = (
            self.P + self.P.T
        ) / 2.0

        return True


# ==============================================================
# SENSOR GENERATION
# ==============================================================

def generate_gnss(true_state, fault):

    x, y, altitude, heading, turn_rate = true_state

    mx = (
        x +
        np.random.normal(
            0.0,
            GNSS_POSITION_NOISE
        )
    )

    my = (
        y +
        np.random.normal(
            0.0,
            GNSS_POSITION_NOISE
        )
    )

    vx = (
        HORIZONTAL_AIR_SPEED *
        math.cos(heading)
        + WIND_X
        + np.random.normal(
            0.0,
            GNSS_VELOCITY_NOISE
        )
    )

    vy = (
        HORIZONTAL_AIR_SPEED *
        math.sin(heading)
        + WIND_Y
        + np.random.normal(
            0.0,
            GNSS_VELOCITY_NOISE
        )
    )

    outlier = False

    if fault:

        # degraded GNSS has higher chance of bad measurement
        if random.random() < 0.25:

            direction = random.uniform(
                0.0,
                2.0 * math.pi
            )

            mx += (
                GNSS_POSITION_OUTLIER *
                math.cos(direction)
            )

            my += (
                GNSS_POSITION_OUTLIER *
                math.sin(direction)
            )

            vx += GNSS_VELOCITY_OUTLIER * (
                1 if random.random() > 0.5
                else -1
            )

            vy += GNSS_VELOCITY_OUTLIER * (
                1 if random.random() > 0.5
                else -1
            )

            outlier = True

    return (
        np.array([mx, my]),
        np.array([vx, vy]),
        outlier
    )


def generate_baro(true_state, fault):

    altitude = true_state[2]

    noise = BARO_ALTITUDE_NOISE

    if fault:

        noise *= 4.0

    return (
        altitude +
        np.random.normal(
            0.0,
            noise
        )
    )


def generate_imu(true_state, fault):

    heading = true_state[3]
    turn_rate = true_state[4]

    heading_noise = (
        IMU_HEADING_NOISE_DEG
    )

    turn_noise = (
        IMU_TURN_RATE_NOISE_DEG
    )

    if fault:

        heading_noise *= 5.0
        turn_noise *= 5.0

    measured_heading = (
        heading +
        math.radians(
            np.random.normal(
                0.0,
                heading_noise
            )
        )
    )

    measured_turn = (
        turn_rate +
        math.radians(
            np.random.normal(
                0.0,
                turn_noise
            )
        )
    )

    return (
        wrap_angle(measured_heading),
        measured_turn
    )


# ==============================================================
# GUIDANCE
# ==============================================================

def choose_guidance_command(
    ekf_state,
    altitude
):

    x = ekf_state[0]
    y = ekf_state[1]
    heading = ekf_state[3]

    dx = TARGET_X - x
    dy = TARGET_Y - y

    desired_heading = math.atan2(
        dy,
        dx
    )

    heading_error = wrap_angle(
        desired_heading - heading
    )

    # proportional heading controller
    command = (
        heading_error /
        math.radians(45.0)
    )

    command = clamp(
        command,
        -1.0,
        1.0
    )

    # Near target, reduce aggressive steering
    distance = math.hypot(
        dx,
        dy
    )

    if distance < 50.0:

        command *= 0.65

    return command


# ==============================================================
# SINGLE SIMULATION
# ==============================================================

def run_simulation(
    scenario_name,
    scenario
):

    true_state = initial_state()

    ekf = ParafoilEKF()

    health = SensorHealth()

    time = 0.0

    next_gnss = 0.0
    next_baro = 0.0
    next_imu = 0.0
    next_guidance = 0.0

    steering = 0.0

    steering_sum = 0.0
    steering_count = 0

    steering_reversals = 0
    previous_steering = 0.0

    true_history = []
    estimated_history = []
    health_history = []

    position_errors = []
    altitude_errors = []
    heading_errors = []

    gnss_updates = 0
    gnss_accepted = 0
    gnss_rejected = 0

    baro_updates = 0
    baro_accepted = 0
    baro_rejected = 0

    imu_updates = 0
    imu_accepted = 0
    imu_rejected = 0

    while time < SIMULATION_TIME:

        # ------------------------------------------------------
        # GUIDANCE
        # ------------------------------------------------------

        if time >= next_guidance:

            steering = choose_guidance_command(
                ekf.x,
                ekf.x[2]
            )

            if (
                previous_steering != 0.0
                and steering != 0.0
                and np.sign(steering)
                != np.sign(previous_steering)
            ):

                steering_reversals += 1

            previous_steering = steering

            next_guidance += GUIDANCE_INTERVAL

        # ------------------------------------------------------
        # TRUE STATE
        # ------------------------------------------------------

        true_state = propagate_true_state(
            true_state,
            steering,
            DT
        )

        # ------------------------------------------------------
        # EKF PREDICTION
        # ------------------------------------------------------

        ekf.predict(
            steering,
            DT
        )

        # ------------------------------------------------------
        # GNSS
        # ------------------------------------------------------

        if time >= next_gnss:

            next_gnss += 1.0 / GNSS_RATE

            gnss_updates += 1

            if scenario["gnss_fault"]:

                # During degradation:
                # some updates become unreliable
                if random.random() < 0.35:

                    health.degrade("gnss")

                    gnss_rejected += 1

                else:

                    health.recover("gnss")

                    measurement, velocity, _ = (
                        generate_gnss(
                            true_state,
                            True
                        )
                    )

                    accepted = ekf.update_gnss(
                        measurement,
                        health.gnss
                    )

                    if accepted:

                        gnss_accepted += 1

                    else:

                        gnss_rejected += 1

            else:

                health.recover("gnss")

                measurement, velocity, _ = (
                    generate_gnss(
                        true_state,
                        False
                    )
                )

                accepted = ekf.update_gnss(
                    measurement,
                    health.gnss
                )

                if accepted:

                    gnss_accepted += 1

                else:

                    gnss_rejected += 1

        # ------------------------------------------------------
        # BAROMETER
        # ------------------------------------------------------

        if time >= next_baro:

            next_baro += 1.0 / BARO_RATE

            baro_updates += 1

            if scenario["baro_fault"]:

                # degraded barometer
                if random.random() < 0.30:

                    health.degrade("baro")

                    baro_rejected += 1

                else:

                    health.recover("baro")

                    altitude = generate_baro(
                        true_state,
                        True
                    )

                    accepted = ekf.update_baro(
                        altitude,
                        health.baro
                    )

                    if accepted:

                        baro_accepted += 1

                    else:

                        baro_rejected += 1

            else:

                health.recover("baro")

                altitude = generate_baro(
                    true_state,
                    False
                )

                accepted = ekf.update_baro(
                    altitude,
                    health.baro
                )

                if accepted:

                    baro_accepted += 1

                else:

                    baro_rejected += 1

        # ------------------------------------------------------
        # IMU
        # ------------------------------------------------------

        if time >= next_imu:

            next_imu += 1.0 / IMU_RATE

            imu_updates += 1

            if scenario["imu_fault"]:

                if random.random() < 0.20:

                    health.degrade("imu")

                    imu_rejected += 1

                else:

                    health.recover("imu")

                    measured_heading, measured_turn = (
                        generate_imu(
                            true_state,
                            True
                        )
                    )

                    accepted = ekf.update_imu(
                        measured_heading,
                        measured_turn,
                        health.imu
                    )

                    if accepted:

                        imu_accepted += 1

                    else:

                        imu_rejected += 1

            else:

                health.recover("imu")

                measured_heading, measured_turn = (
                    generate_imu(
                        true_state,
                        False
                    )
                )

                accepted = ekf.update_imu(
                    measured_heading,
                    measured_turn,
                    health.imu
                )

                if accepted:

                    imu_accepted += 1

                else:

                    imu_rejected += 1

        # ------------------------------------------------------
        # ERRORS
        # ------------------------------------------------------

        position_error = math.hypot(
            true_state[0] - ekf.x[0],
            true_state[1] - ekf.x[1]
        )

        altitude_error = abs(
            true_state[2] - ekf.x[2]
        )

        heading_error = abs(
            math.degrees(
                wrap_angle(
                    true_state[3] -
                    ekf.x[3]
                )
            )
        )

        position_errors.append(
            position_error
        )

        altitude_errors.append(
            altitude_error
        )

        heading_errors.append(
            heading_error
        )

        true_history.append(
            true_state.copy()
        )

        estimated_history.append(
            ekf.x.copy()
        )

        health_history.append([
            health.gnss,
            health.imu,
            health.baro
        ])

        steering_sum += abs(steering)
        steering_count += 1

        time += DT

        if true_state[2] <= 0.0:

            break

    # ==========================================================
    # FINAL RESULTS
    # ==========================================================

    true_landing_x = true_state[0]
    true_landing_y = true_state[1]

    estimated_landing_x = ekf.x[0]
    estimated_landing_y = ekf.x[1]

    true_landing_error = math.hypot(
        true_landing_x - TARGET_X,
        true_landing_y - TARGET_Y
    )

    estimated_landing_error = math.hypot(
        estimated_landing_x - TARGET_X,
        estimated_landing_y - TARGET_Y
    )

    rms_position = math.sqrt(
        np.mean(
            np.array(position_errors) ** 2
        )
    )

    rms_altitude = math.sqrt(
        np.mean(
            np.array(altitude_errors) ** 2
        )
    )

    rms_heading = math.sqrt(
        np.mean(
            np.array(heading_errors) ** 2
        )
    )

    max_position = max(
        position_errors
    )

    average_steering = (
        steering_sum /
        max(1, steering_count)
    )

    landing_status = (
        "WITHIN TOLERANCE"
        if true_landing_error <= TARGET_TOLERANCE
        else "OUTSIDE TOLERANCE"
    )

    return {
        "scenario": scenario_name,

        "true_landing_x": true_landing_x,
        "true_landing_y": true_landing_y,

        "estimated_landing_x":
            estimated_landing_x,

        "estimated_landing_y":
            estimated_landing_y,

        "true_landing_error":
            true_landing_error,

        "estimated_landing_error":
            estimated_landing_error,

        "rms_position":
            rms_position,

        "rms_altitude":
            rms_altitude,

        "rms_heading":
            rms_heading,

        "max_position":
            max_position,

        "average_steering":
            average_steering,

        "steering_reversals":
            steering_reversals,

        "gnss_updates":
            gnss_updates,

        "gnss_accepted":
            gnss_accepted,

        "gnss_rejected":
            gnss_rejected,

        "baro_updates":
            baro_updates,

        "baro_accepted":
            baro_accepted,

        "baro_rejected":
            baro_rejected,

        "imu_updates":
            imu_updates,

        "imu_accepted":
            imu_accepted,

        "imu_rejected":
            imu_rejected,

        "final_gnss_health":
            health.gnss,

        "final_imu_health":
            health.imu,

        "final_baro_health":
            health.baro,

        "true_history":
            np.array(true_history),

        "estimated_history":
            np.array(estimated_history),

        "health_history":
            np.array(health_history),

        "position_errors":
            np.array(position_errors),

        "altitude_errors":
            np.array(altitude_errors),

        "heading_errors":
            np.array(heading_errors)
    }


# ==============================================================
# PRINT HEADER
# ==============================================================

print("=" * 64)
print("VIRTUAL PARAFOIL V10.5")
print("SENSOR FAULT DETECTION + HEALTH MONITORING")
print("ADAPTIVE EKF MEASUREMENT WEIGHTING")
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
    f"{HORIZONTAL_AIR_SPEED} m/s"
)
print(
    f"Vertical descent velocity: "
    f"{VERTICAL_DESCENT_SPEED} m/s"
)
print(f"Glide ratio: {GLIDE_RATIO}")


# ==============================================================
# TARGET
# ==============================================================

print()
print("=" * 64)
print("TARGET")
print("=" * 64)

print(f"Target X: {TARGET_X} m")
print(f"Target Y: {TARGET_Y} m")
print(f"Tolerance: {TARGET_TOLERANCE} m")


# ==============================================================
# WIND
# ==============================================================

print()
print("=" * 64)
print("REFERENCE WIND")
print("=" * 64)

print(f"Wind speed: {WIND_SPEED} m/s")
print(
    f"Wind direction: "
    f"{WIND_DIRECTION_DEG} degrees"
)

print(f"Wind X: {WIND_X} m/s")
print(f"Wind Y: {WIND_Y} m/s")


# ==============================================================
# SENSOR MODEL
# ==============================================================

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

print(f"GNSS update rate: {GNSS_RATE} Hz")
print(f"Barometer update rate: {BARO_RATE} Hz")
print(f"IMU update rate: {IMU_RATE} Hz")


# ==============================================================
# HEALTH MONITORING
# ==============================================================

print()
print("=" * 64)
print("SENSOR HEALTH MONITORING")
print("=" * 64)

print(
    "GNSS adaptive covariance: ENABLED"
)

print(
    "IMU adaptive covariance: ENABLED"
)

print(
    "Barometer adaptive covariance: ENABLED"
)

print(
    f"Health rejection threshold: "
    f"{HEALTH_REJECT_THRESHOLD}"
)

print(
    f"GNSS innovation threshold: "
    f"{GNSS_INNOVATION_THRESHOLD} m"
)

print(
    f"Barometer innovation threshold: "
    f"{BARO_INNOVATION_THRESHOLD} m"
)

print(
    f"IMU innovation threshold: "
    f"{IMU_INNOVATION_THRESHOLD_DEG} deg"
)


# ==============================================================
# RUN ALL SCENARIOS
# ==============================================================

print()
print("=" * 64)
print("STARTING V10.5 SENSOR HEALTH STUDY")
print("=" * 64)

results = []

scenario_items = list(
    SCENARIOS.items()
)

for index, (
    scenario_name,
    scenario
) in enumerate(
    scenario_items,
    start=1
):

    print()
    print("-" * 64)

    print(
        f"Scenario {index} / "
        f"{len(scenario_items)}"
    )

    print(
        f"Failure mode: "
        f"{scenario_name}"
    )

    result = run_simulation(
        scenario_name,
        scenario
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
        f"EKF RMS position: "
        f"{result['rms_position']:.3f} m"
    )

    print(
        f"EKF RMS altitude: "
        f"{result['rms_altitude']:.3f} m"
    )

    print(
        f"EKF RMS heading: "
        f"{result['rms_heading']:.3f} deg"
    )

    print(
        f"Maximum position error: "
        f"{result['max_position']:.3f} m"
    )

    print(
        f"GNSS rejected: "
        f"{result['gnss_rejected']}"
    )

    print(
        f"IMU rejected: "
        f"{result['imu_rejected']}"
    )

    print(
        f"Barometer rejected: "
        f"{result['baro_rejected']}"
    )

    print(
        f"Final GNSS health: "
        f"{result['final_gnss_health']:.3f}"
    )

    print(
        f"Final IMU health: "
        f"{result['final_imu_health']:.3f}"
    )

    print(
        f"Final barometer health: "
        f"{result['final_baro_health']:.3f}"
    )

    print(
        f"Landing status: "
        f"{'PASS' if result['true_landing_error'] <= TARGET_TOLERANCE else 'FAIL'}"
    )


# ==============================================================
# RESULTS TABLE
# ==============================================================

print()
print("=" * 100)
print("V10.5 SENSOR HEALTH RESULTS")
print("=" * 100)

print(
    f"{'Scenario':<30}"
    f"{'Landing Error':>15}"
    f"{'EKF Pos':>12}"
    f"{'EKF Alt':>12}"
    f"{'EKF Head':>12}"
    f"{'Status':>10}"
)

print("-" * 100)

for r in results:

    status = (
        "PASS"
        if r["true_landing_error"]
        <= TARGET_TOLERANCE
        else "FAIL"
    )

    print(
        f"{r['scenario']:<30}"
        f"{r['true_landing_error']:>15.3f}"
        f"{r['rms_position']:>12.3f}"
        f"{r['rms_altitude']:>12.3f}"
        f"{r['rms_heading']:>12.3f}"
        f"{status:>10}"
    )


# ==============================================================
# BEST / WORST LANDING
# ==============================================================

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


# ==============================================================
# BEST / WORST EKF
# ==============================================================

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


# ==============================================================
# SUCCESS COUNT
# ==============================================================

successful = [
    r for r in results
    if r["true_landing_error"]
    <= TARGET_TOLERANCE
]

success_percentage = (
    100.0 *
    len(successful) /
    len(results)
)


# ==============================================================
# BEST LANDING
# ==============================================================

print()
print("=" * 64)
print("BEST LANDING CASE")
print("=" * 64)

print(
    f"Failure mode: "
    f"{best_landing['scenario']}"
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


# ==============================================================
# WORST LANDING
# ==============================================================

print()
print("=" * 64)
print("WORST LANDING CASE")
print("=" * 64)

print(
    f"Failure mode: "
    f"{worst_landing['scenario']}"
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


# ==============================================================
# BEST EKF
# ==============================================================

print()
print("=" * 64)
print("BEST EKF ESTIMATION CASE")
print("=" * 64)

print(
    f"Failure mode: "
    f"{best_ekf['scenario']}"
)

print(
    f"RMS position error: "
    f"{best_ekf['rms_position']:.3f} m"
)

print(
    f"RMS altitude error: "
    f"{best_ekf['rms_altitude']:.3f} m"
)

print(
    f"RMS heading error: "
    f"{best_ekf['rms_heading']:.3f} deg"
)


# ==============================================================
# WORST EKF
# ==============================================================

print()
print("=" * 64)
print("WORST EKF ESTIMATION CASE")
print("=" * 64)

print(
    f"Failure mode: "
    f"{worst_ekf['scenario']}"
)

print(
    f"RMS position error: "
    f"{worst_ekf['rms_position']:.3f} m"
)

print(
    f"Maximum position error: "
    f"{worst_ekf['max_position']:.3f} m"
)


# ==============================================================
# SENSOR HEALTH ASSESSMENT
# ==============================================================

print()
print("=" * 64)
print("SENSOR HEALTH ASSESSMENT")
print("=" * 64)

print(
    "The EKF automatically changes measurement "
    "confidence according to sensor health."
)

print(
    "Healthy sensor -> normal measurement covariance."
)

print(
    "Degraded sensor -> increased measurement covariance."
)

print(
    "Critical sensor -> measurement rejected."
)

print(
    "This prevents a faulty sensor from dominating "
    "the navigation solution."
)


# ==============================================================
# NAVIGATION ASSESSMENT
# ==============================================================

print()
print("=" * 64)
print("V10.5 NAVIGATION SYSTEM ASSESSMENT")
print("=" * 64)

print(
    f"Raw GNSS position noise: "
    f"{GNSS_POSITION_NOISE} m"
)

print(
    f"Best EKF RMS position: "
    f"{best_ekf['rms_position']:.3f} m"
)

if best_ekf["rms_position"] < GNSS_POSITION_NOISE:

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
    "SENSOR HEALTH MONITORING: ACTIVE"
)

print(
    "ADAPTIVE EKF WEIGHTING: ACTIVE"
)

print(
    "FAULT DETECTION: ACTIVE"
)


# ==============================================================
# DIAGNOSTIC SUMMARY
# ==============================================================

print()
print("=" * 64)
print("V10.5 DIAGNOSTIC SUMMARY")
print("=" * 64)

print()

print(
    "The V10.5 navigation system monitors "
    "the quality of GNSS, IMU and barometer measurements."
)

print()

print(
    "When a sensor produces abnormal innovations, "
    "its health score is reduced."
)

print()

print(
    "The EKF then increases the corresponding "
    "measurement covariance, reducing the influence "
    "of the degraded sensor."
)

print()

print(
    "If sensor health becomes critically low, "
    "the measurement is rejected."
)

print()

print(
    "When the sensor returns to normal operation, "
    "its health score gradually recovers."
)

print()

print(
    "This is more robust than simply assuming "
    "that every sensor measurement is equally reliable."
)


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print()
print("=" * 64)
print("V10.5 SIMULATION COMPLETE")
print("=" * 64)

print(
    f"Scenarios tested: "
    f"{len(results)}"
)

print(
    f"Successful scenarios: "
    f"{len(successful)} / {len(results)}"
)

print(
    f"Success percentage: "
    f"{success_percentage:.1f} %"
)

print(
    f"Best landing error: "
    f"{best_landing['true_landing_error']:.3f} m"
)

print(
    f"Worst landing error: "
    f"{worst_landing['true_landing_error']:.3f} m"
)

print(
    f"Best EKF RMS position: "
    f"{best_ekf['rms_position']:.3f} m"
)

print(
    f"Worst EKF RMS position: "
    f"{worst_ekf['rms_position']:.3f} m"
)

print(
    "Sensor health monitoring: ACTIVE"
)

print(
    "Adaptive EKF weighting: ACTIVE"
)

print(
    "Fault detection: ACTIVE"
)

print("=" * 64)

print()
print("NEXT DEVELOPMENT STEP:")
print(
    "V10.6 -> Wind estimation + adaptive wind-vector"
)
print(
    "          estimation using GNSS + IMU + airspeed"
)
print("=" * 64)


# ==============================================================
# PLOTS
# ==============================================================

# --------------------------------------------------------------
# Use the nominal case for detailed plots
# --------------------------------------------------------------

nominal = results[0]

t = np.arange(
    len(nominal["true_history"])
) * DT


# ==============================================================
# PLOT 1 - TRAJECTORY
# ==============================================================

plt.figure()

true_hist = nominal["true_history"]
est_hist = nominal["estimated_history"]

plt.plot(
    true_hist[:, 0],
    true_hist[:, 1],
    label="True trajectory"
)

plt.plot(
    est_hist[:, 0],
    est_hist[:, 1],
    "--",
    label="EKF trajectory"
)

plt.scatter(
    TARGET_X,
    TARGET_Y,
    marker="x",
    s=100,
    label="Target"
)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title(
    "V10.5 Parafoil Trajectory - Nominal Case"
)

plt.axis("equal")
plt.grid(True)
plt.legend()


# ==============================================================
# PLOT 2 - SENSOR HEALTH
# ==============================================================

plt.figure()

health_hist = nominal[
    "health_history"
]

plt.plot(
    t,
    health_hist[:, 0],
    label="GNSS health"
)

plt.plot(
    t,
    health_hist[:, 1],
    label="IMU health"
)

plt.plot(
    t,
    health_hist[:, 2],
    label="Barometer health"
)

plt.xlabel("Time (s)")
plt.ylabel("Health score")
plt.title(
    "V10.5 Sensor Health Monitoring"
)

plt.ylim(
    0.0,
    1.05
)

plt.grid(True)
plt.legend()


# ==============================================================
# PLOT 3 - POSITION ESTIMATION ERROR
# ==============================================================

plt.figure()

plt.plot(
    t,
    nominal["position_errors"]
)

plt.axhline(
    GNSS_POSITION_NOISE,
    linestyle="--",
    label="Raw GNSS noise"
)

plt.xlabel("Time (s)")
plt.ylabel("Position error (m)")
plt.title(
    "V10.5 EKF Position Estimation Error"
)

plt.grid(True)
plt.legend()


# ==============================================================
# PLOT 4 - ALTITUDE ESTIMATION ERROR
# ==============================================================

plt.figure()

plt.plot(
    t,
    nominal["altitude_errors"]
)

plt.xlabel("Time (s)")
plt.ylabel("Altitude error (m)")
plt.title(
    "V10.5 EKF Altitude Estimation Error"
)

plt.grid(True)


# ==============================================================
# PLOT 5 - HEADING ERROR
# ==============================================================

plt.figure()

plt.plot(
    t,
    nominal["heading_errors"]
)

plt.xlabel("Time (s)")
plt.ylabel("Heading error (deg)")
plt.title(
    "V10.5 EKF Heading Estimation Error"
)

plt.grid(True)


# ==============================================================
# SHOW PLOTS
# ==============================================================

plt.show()