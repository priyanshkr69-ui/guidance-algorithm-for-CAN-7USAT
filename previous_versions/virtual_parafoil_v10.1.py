import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# VIRTUAL PARAFOIL V10.1
# EKF TUNING + SENSOR UPDATE-RATE STUDY
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V10.1")
print("EKF TUNING + SENSOR UPDATE-RATE STUDY")
print("========================================")


# ============================================================
# PARAFOIL PARAMETERS
# ============================================================

mass = 1.0
g = 9.81

span = 1.6
chord = 0.6
area = span * chord

rho = 1.225

CL = 0.40
CD = 0.25

weight = mass * g

glide_angle = np.arctan(CD / CL)

airspeed = np.sqrt(
    weight /
    (
        0.5 *
        rho *
        area *
        CL *
        np.cos(glide_angle)
    )
)

horizontal_air_velocity = (
    airspeed * np.cos(glide_angle)
)

vertical_velocity = (
    airspeed * np.sin(glide_angle)
)

glide_ratio = (
    horizontal_air_velocity /
    vertical_velocity
)


# ============================================================
# TARGET
# ============================================================

target_x = 500.0
target_y = 200.0

target_tolerance = 20.0


# ============================================================
# REFERENCE WIND
# ============================================================

wind_speed = 3.0
wind_direction_deg = 0.0

wind_direction = np.radians(wind_direction_deg)

wind_x = (
    wind_speed *
    np.cos(wind_direction)
)

wind_y = (
    wind_speed *
    np.sin(wind_direction)
)


# ============================================================
# GUIDANCE PARAMETERS
# ============================================================

guidance_interval = 2.0

max_turn_rate = np.radians(15.0)

candidate_commands = np.linspace(
    -1.0,
    1.0,
    21
)


# ============================================================
# ADAPTIVE PREDICTION HORIZON
# ============================================================

def get_prediction_horizon(altitude):

    if altitude > 400.0:
        return 20.0

    elif altitude > 200.0:
        return 15.0

    elif altitude > 100.0:
        return 10.0

    else:
        return 5.0


# ============================================================
# SENSOR PARAMETERS
# ============================================================

gnss_position_noise = 3.0
gnss_velocity_noise = 0.3

barometer_altitude_noise = 2.0

imu_heading_noise = np.radians(2.0)
imu_turn_rate_noise = np.radians(0.5)

imu_update_rate = 50.0


# ============================================================
# SIMULATION PARAMETERS
# ============================================================

dt = 1.0 / imu_update_rate

simulation_time = 200.0

random_seed = 42


# ============================================================
# PRINT PARAMETERS
# ============================================================

print()
print("========================================")
print("PARAFOIL PARAMETERS")
print("========================================")

print("Area:", area, "m^2")
print("Mass:", mass, "kg")
print("CL:", CL)
print("CD:", CD)

print("Airspeed:", airspeed, "m/s")
print(
    "Horizontal air velocity:",
    horizontal_air_velocity,
    "m/s"
)

print(
    "Vertical descent velocity:",
    vertical_velocity,
    "m/s"
)

print("Glide ratio:", glide_ratio)


print()
print("========================================")
print("TARGET")
print("========================================")

print("Target X:", target_x, "m")
print("Target Y:", target_y, "m")
print("Tolerance:", target_tolerance, "m")


print()
print("========================================")
print("REFERENCE WIND")
print("========================================")

print("Wind speed:", wind_speed, "m/s")
print(
    "Wind direction:",
    wind_direction_deg,
    "degrees"
)

print("Wind X:", wind_x, "m/s")
print("Wind Y:", wind_y, "m/s")


print()
print("========================================")
print("SENSOR MODEL")
print("========================================")

print(
    "GNSS position noise:",
    gnss_position_noise,
    "m"
)

print(
    "GNSS velocity noise:",
    gnss_velocity_noise,
    "m/s"
)

print(
    "Barometer altitude noise:",
    barometer_altitude_noise,
    "m"
)

print(
    "IMU heading noise:",
    np.degrees(imu_heading_noise),
    "deg"
)

print(
    "IMU turn-rate noise:",
    np.degrees(imu_turn_rate_noise),
    "deg/s"
)

print(
    "IMU update rate:",
    imu_update_rate,
    "Hz"
)


# ============================================================
# EKF
#
# STATE:
#
# x = [
#     position_x,
#     position_y,
#     altitude,
#     velocity_x,
#     velocity_y,
#     vertical_velocity,
#     heading,
#     turn_rate
# ]
# ============================================================

STATE_SIZE = 8


def normalize_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


class ParafoilEKF:

    def __init__(self):

        # ----------------------------------------------------
        # Initial state
        # ----------------------------------------------------

        self.x = np.array([
            0.0,
            0.0,
            600.0,
            horizontal_air_velocity + wind_x,
            wind_y,
            vertical_velocity,
            0.0,
            0.0
        ], dtype=float)

        # ----------------------------------------------------
        # Initial covariance
        # ----------------------------------------------------

        self.P = np.diag([
            25.0,
            25.0,
            16.0,
            1.0,
            1.0,
            1.0,
            np.radians(10.0) ** 2,
            np.radians(2.0) ** 2
        ])

        # ----------------------------------------------------
        # Process noise
        # ----------------------------------------------------

        self.Q = np.diag([
            0.05,
            0.05,
            0.10,
            0.20,
            0.20,
            0.10,
            np.radians(0.2) ** 2,
            np.radians(0.5) ** 2
        ])


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        steering_command,
        dt
    ):

        px = self.x[0]
        py = self.x[1]
        altitude = self.x[2]

        vx = self.x[3]
        vy = self.x[4]

        vz = self.x[5]

        heading = self.x[6]
        turn_rate = self.x[7]

        # ----------------------------------------------------
        # Parafoil turn dynamics
        # ----------------------------------------------------

        commanded_turn_rate = (
            max_turn_rate *
            steering_command
        )

        turn_rate += (
            commanded_turn_rate -
            turn_rate
        ) * 0.35

        heading += (
            turn_rate *
            dt
        )

        heading = normalize_angle(
            heading
        )

        # ----------------------------------------------------
        # Air velocity
        # ----------------------------------------------------

        vx_air = (
            horizontal_air_velocity *
            np.cos(heading)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(heading)
        )

        # ----------------------------------------------------
        # Ground velocity
        # ----------------------------------------------------

        vx = vx_air + wind_x
        vy = vy_air + wind_y

        # ----------------------------------------------------
        # Vertical velocity
        # ----------------------------------------------------

        vz = vertical_velocity

        # ----------------------------------------------------
        # Position update
        # ----------------------------------------------------

        px += vx * dt
        py += vy * dt

        altitude -= vz * dt

        altitude = max(
            altitude,
            0.0
        )

        # ----------------------------------------------------
        # State update
        # ----------------------------------------------------

        self.x = np.array([
            px,
            py,
            altitude,
            vx,
            vy,
            vz,
            heading,
            turn_rate
        ])

        # ----------------------------------------------------
        # State transition Jacobian
        # ----------------------------------------------------

        F = np.eye(
            STATE_SIZE
        )

        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = -dt

        F[6, 7] = dt

        self.P = (
            F @
            self.P @
            F.T
            +
            self.Q
        )

        self.P = (
            self.P +
            self.P.T
        ) / 2.0


    # ========================================================
    # GNSS POSITION UPDATE
    # ========================================================

    def update_gnss_position(
        self,
        measurement
    ):

        z = np.array([
            measurement[0],
            measurement[1]
        ])

        H = np.zeros(
            (2, STATE_SIZE)
        )

        H[0, 0] = 1.0
        H[1, 1] = 1.0

        R = np.diag([
            gnss_position_noise ** 2,
            gnss_position_noise ** 2
        ])

        innovation = (
            z -
            H @ self.x
        )

        S = (
            H @
            self.P @
            H.T
            +
            R
        )

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x += (
            K @ innovation
        )

        I = np.eye(
            STATE_SIZE
        )

        self.P = (
            I -
            K @ H
        ) @ self.P

        self.x[6] = normalize_angle(
            self.x[6]
        )


    # ========================================================
    # GNSS VELOCITY UPDATE
    # ========================================================

    def update_gnss_velocity(
        self,
        measurement
    ):

        z = np.array([
            measurement[0],
            measurement[1]
        ])

        H = np.zeros(
            (2, STATE_SIZE)
        )

        H[0, 3] = 1.0
        H[1, 4] = 1.0

        R = np.diag([
            gnss_velocity_noise ** 2,
            gnss_velocity_noise ** 2
        ])

        innovation = (
            z -
            H @ self.x
        )

        S = (
            H @
            self.P @
            H.T
            +
            R
        )

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x += (
            K @ innovation
        )

        I = np.eye(
            STATE_SIZE
        )

        self.P = (
            I -
            K @ H
        ) @ self.P


    # ========================================================
    # BAROMETER UPDATE
    # ========================================================

    def update_barometer(
        self,
        measurement
    ):

        z = measurement

        H = np.zeros(
            (1, STATE_SIZE)
        )

        H[0, 2] = 1.0

        R = np.array([
            [barometer_altitude_noise ** 2]
        ])

        innovation = (
            z -
            H @ self.x
        )

        S = (
            H @
            self.P @
            H.T
            +
            R
        )

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x += (
            (
                K @ innovation
            ).flatten()
        )

        I = np.eye(
            STATE_SIZE
        )

        self.P = (
            I -
            K @ H
        ) @ self.P


    # ========================================================
    # IMU HEADING UPDATE
    # ========================================================

    def update_heading(
        self,
        measurement
    ):

        innovation = normalize_angle(
            measurement -
            self.x[6]
        )

        H = np.zeros(
            (1, STATE_SIZE)
        )

        H[0, 6] = 1.0

        R = np.array([
            [imu_heading_noise ** 2]
        ])

        S = (
            H @
            self.P @
            H.T
            +
            R
        )

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x += (
            (
                K *
                innovation
            ).flatten()
        )

        I = np.eye(
            STATE_SIZE
        )

        self.P = (
            I -
            K @ H
        ) @ self.P

        self.x[6] = normalize_angle(
            self.x[6]
        )


# ============================================================
# GUIDANCE PREDICTION
# ============================================================

def predict_landing(
    current_x,
    current_y,
    current_altitude,
    current_heading,
    steering_command,
    wind_x_local,
    wind_y_local,
    prediction_horizon
):

    px = current_x
    py = current_y

    pheading = current_heading

    prediction_dt = 0.25

    steps = max(
        1,
        int(
            prediction_horizon /
            prediction_dt
        )
    )

    for _ in range(steps):

        turn_rate = (
            max_turn_rate *
            steering_command
        )

        pheading += (
            turn_rate *
            prediction_dt
        )

        pheading = normalize_angle(
            pheading
        )

        vx_air = (
            horizontal_air_velocity *
            np.cos(pheading)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(pheading)
        )

        vx_ground = (
            vx_air +
            wind_x_local
        )

        vy_ground = (
            vy_air +
            wind_y_local
        )

        px += (
            vx_ground *
            prediction_dt
        )

        py += (
            vy_ground *
            prediction_dt
        )

    return px, py


# ============================================================
# SELECT GUIDANCE COMMAND
# ============================================================

def select_guidance_command(
    estimated_state,
    wind_x_local,
    wind_y_local
):

    current_x = estimated_state[0]
    current_y = estimated_state[1]
    current_altitude = estimated_state[2]
    current_heading = estimated_state[6]

    prediction_horizon = (
        get_prediction_horizon(
            current_altitude
        )
    )

    best_command = 0.0
    best_cost = float("inf")

    # --------------------------------------------------------
    # Heading toward target
    # --------------------------------------------------------

    target_heading = np.arctan2(
        target_y - current_y,
        target_x - current_x
    )

    heading_error = normalize_angle(
        target_heading -
        current_heading
    )

    for command in candidate_commands:

        predicted_x, predicted_y = (
            predict_landing(
                current_x,
                current_y,
                current_altitude,
                current_heading,
                command,
                wind_x_local,
                wind_y_local,
                prediction_horizon
            )
        )

        position_error = np.sqrt(
            (
                predicted_x -
                target_x
            ) ** 2
            +
            (
                predicted_y -
                target_y
            ) ** 2
        )

        # ----------------------------------------------------
        # Estimate final heading
        # ----------------------------------------------------

        predicted_heading = (
            current_heading +
            (
                max_turn_rate *
                command *
                prediction_horizon
            )
        )

        predicted_heading = normalize_angle(
            predicted_heading
        )

        final_heading_error = abs(
            normalize_angle(
                target_heading -
                predicted_heading
            )
        )

        # ----------------------------------------------------
        # Combined guidance cost
        # ----------------------------------------------------

        heading_cost = (
            final_heading_error /
            np.pi
        ) * 50.0

        cost = (
            0.75 *
            position_error
            +
            0.25 *
            heading_cost
        )

        if cost < best_cost:

            best_cost = cost
            best_command = command

    return best_command


# ============================================================
# SENSOR RATE STUDY
# ============================================================

gnss_rates = [
    1.0,
    2.0,
    5.0,
    10.0
]

barometer_rates = [
    5.0,
    10.0,
    20.0
]


# ============================================================
# RUN ONE SIMULATION
# ============================================================

def run_simulation(
    gnss_rate,
    barometer_rate,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # TRUE STATE
    # --------------------------------------------------------

    altitude = 600.0

    x = 0.0
    y = 0.0

    heading = 0.0

    time = 0.0

    current_steering = 0.0

    next_guidance_update = 0.0

    next_gnss_update = 0.0

    next_barometer_update = 0.0

    next_imu_update = 0.0

    # --------------------------------------------------------
    # EKF
    # --------------------------------------------------------

    ekf = ParafoilEKF()

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    true_x_history = []
    true_y_history = []

    estimated_x_history = []
    estimated_y_history = []

    true_altitude_history = []
    estimated_altitude_history = []

    true_heading_history = []
    estimated_heading_history = []

    position_errors = []
    altitude_errors = []
    heading_errors = []

    steering_history = []

    # --------------------------------------------------------
    # GNSS velocity state
    # --------------------------------------------------------

    true_vx = (
        horizontal_air_velocity +
        wind_x
    )

    true_vy = wind_y

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while altitude > 0.0:

        # ----------------------------------------------------
        # TRUE PARAFOIL DYNAMICS
        # ----------------------------------------------------

        turn_rate = (
            max_turn_rate *
            current_steering
        )

        heading += (
            turn_rate *
            dt
        )

        heading = normalize_angle(
            heading
        )

        vx_air = (
            horizontal_air_velocity *
            np.cos(heading)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(heading)
        )

        true_vx = (
            vx_air +
            wind_x
        )

        true_vy = (
            vy_air +
            wind_y
        )

        x += (
            true_vx *
            dt
        )

        y += (
            true_vy *
            dt
        )

        altitude -= (
            vertical_velocity *
            dt
        )

        altitude = max(
            altitude,
            0.0
        )

        # ----------------------------------------------------
        # EKF PREDICTION
        # ----------------------------------------------------

        ekf.predict(
            current_steering,
            dt
        )

        # ----------------------------------------------------
        # GNSS UPDATE
        # ----------------------------------------------------

        if time >= next_gnss_update:

            gnss_x = (
                x +
                rng.normal(
                    0.0,
                    gnss_position_noise
                )
            )

            gnss_y = (
                y +
                rng.normal(
                    0.0,
                    gnss_position_noise
                )
            )

            ekf.update_gnss_position(
                [gnss_x, gnss_y]
            )

            gnss_vx = (
                true_vx +
                rng.normal(
                    0.0,
                    gnss_velocity_noise
                )
            )

            gnss_vy = (
                true_vy +
                rng.normal(
                    0.0,
                    gnss_velocity_noise
                )
            )

            ekf.update_gnss_velocity(
                [gnss_vx, gnss_vy]
            )

            next_gnss_update += (
                1.0 /
                gnss_rate
            )

        # ----------------------------------------------------
        # BAROMETER UPDATE
        # ----------------------------------------------------

        if time >= next_barometer_update:

            baro_altitude = (
                altitude +
                rng.normal(
                    0.0,
                    barometer_altitude_noise
                )
            )

            ekf.update_barometer(
                baro_altitude
            )

            next_barometer_update += (
                1.0 /
                barometer_rate
            )

        # ----------------------------------------------------
        # IMU UPDATE
        # ----------------------------------------------------

        if time >= next_imu_update:

            imu_heading = (
                heading +
                rng.normal(
                    0.0,
                    imu_heading_noise
                )
            )

            imu_heading = normalize_angle(
                imu_heading
            )

            ekf.update_heading(
                imu_heading
            )

            next_imu_update += (
                1.0 /
                imu_update_rate
            )

        # ----------------------------------------------------
        # GUIDANCE UPDATE
        # ----------------------------------------------------

        if time >= next_guidance_update:

            current_steering = (
                select_guidance_command(
                    ekf.x,
                    wind_x,
                    wind_y
                )
            )

            next_guidance_update += (
                guidance_interval
            )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        true_x_history.append(x)
        true_y_history.append(y)

        estimated_x_history.append(
            ekf.x[0]
        )

        estimated_y_history.append(
            ekf.x[1]
        )

        true_altitude_history.append(
            altitude
        )

        estimated_altitude_history.append(
            ekf.x[2]
        )

        true_heading_history.append(
            heading
        )

        estimated_heading_history.append(
            ekf.x[6]
        )

        position_error = np.sqrt(
            (
                ekf.x[0] -
                x
            ) ** 2
            +
            (
                ekf.x[1] -
                y
            ) ** 2
        )

        altitude_error = abs(
            ekf.x[2] -
            altitude
        )

        heading_error = abs(
            np.degrees(
                normalize_angle(
                    ekf.x[6] -
                    heading
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

        steering_history.append(
            current_steering
        )

        time += dt

        if time > simulation_time:

            break

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    true_landing_error = np.sqrt(
        (
            x -
            target_x
        ) ** 2
        +
        (
            y -
            target_y
        ) ** 2
    )

    estimated_landing_error = np.sqrt(
        (
            ekf.x[0] -
            target_x
        ) ** 2
        +
        (
            ekf.x[1] -
            target_y
        ) ** 2
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

    steering_array = np.array(
        steering_history
    )

    mean_position_error = np.mean(
        position_errors
    )

    rms_position_error = np.sqrt(
        np.mean(
            position_errors ** 2
        )
    )

    max_position_error = np.max(
        position_errors
    )

    mean_altitude_error = np.mean(
        altitude_errors
    )

    rms_altitude_error = np.sqrt(
        np.mean(
            altitude_errors ** 2
        )
    )

    mean_heading_error = np.mean(
        heading_errors
    )

    rms_heading_error = np.sqrt(
        np.mean(
            heading_errors ** 2
        )
    )

    average_steering = np.mean(
        np.abs(
            steering_array
        )
    )

    steering_reversals = 0

    for i in range(
        1,
        len(steering_array)
    ):

        if (
            steering_array[i] != 0
            and
            steering_array[i - 1] != 0
            and
            np.sign(
                steering_array[i]
            )
            !=
            np.sign(
                steering_array[i - 1]
            )
        ):

            steering_reversals += 1

    return {
        "gnss_rate": gnss_rate,
        "barometer_rate": barometer_rate,

        "true_x": x,
        "true_y": y,

        "estimated_x": ekf.x[0],
        "estimated_y": ekf.x[1],

        "true_landing_error":
            true_landing_error,

        "estimated_landing_error":
            estimated_landing_error,

        "flight_time": time,

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

        "mean_heading_error":
            mean_heading_error,

        "rms_heading_error":
            rms_heading_error,

        "average_steering":
            average_steering,

        "steering_reversals":
            steering_reversals,

        "true_x_history":
            true_x_history,

        "true_y_history":
            true_y_history,

        "estimated_x_history":
            estimated_x_history,

        "estimated_y_history":
            estimated_y_history,

        "true_altitude_history":
            true_altitude_history,

        "estimated_altitude_history":
            estimated_altitude_history,

        "true_heading_history":
            true_heading_history,

        "estimated_heading_history":
            estimated_heading_history,

        "position_errors":
            position_errors,

        "altitude_errors":
            altitude_errors,

        "heading_errors":
            heading_errors
    }


# ============================================================
# RUN SENSOR RATE STUDY
# ============================================================

all_results = []

print()
print("========================================")
print("V10.1 SENSOR UPDATE-RATE STUDY")
print("========================================")

print()
print(
    "GNSS rates:",
    gnss_rates,
    "Hz"
)

print(
    "Barometer rates:",
    barometer_rates,
    "Hz"
)

print(
    "IMU rate:",
    imu_update_rate,
    "Hz"
)

print()
print("Starting simulations...")


simulation_number = 0

total_simulations = (
    len(gnss_rates) *
    len(barometer_rates)
)


for gnss_rate in gnss_rates:

    for barometer_rate in barometer_rates:

        simulation_number += 1

        print()
        print("----------------------------------------")

        print(
            "Simulation",
            simulation_number,
            "/",
            total_simulations
        )

        print(
            "GNSS:",
            gnss_rate,
            "Hz"
        )

        print(
            "Barometer:",
            barometer_rate,
            "Hz"
        )

        result = run_simulation(
            gnss_rate,
            barometer_rate,
            random_seed +
            simulation_number
        )

        all_results.append(
            result
        )

        print(
            "True landing error:",
            round(
                result[
                    "true_landing_error"
                ],
                3
            ),
            "m"
        )

        print(
            "EKF RMS position error:",
            round(
                result[
                    "rms_position_error"
                ],
                3
            ),
            "m"
        )

        print(
            "EKF RMS altitude error:",
            round(
                result[
                    "rms_altitude_error"
                ],
                3
            ),
            "m"
        )

        print(
            "EKF RMS heading error:",
            round(
                result[
                    "rms_heading_error"
                ],
                3
            ),
            "deg"
        )


# ============================================================
# FIND BEST CONFIGURATION
# ============================================================

best_estimator = min(
    all_results,
    key=lambda r:
    r["rms_position_error"]
)

best_landing = min(
    all_results,
    key=lambda r:
    r["true_landing_error"]
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print()
print()
print("============================================================")
print("V10.1 SENSOR UPDATE-RATE RESULTS")
print("============================================================")

print(
    f"{'GNSS Hz':<12}"
    f"{'Baro Hz':<12}"
    f"{'Landing Error':<18}"
    f"{'EKF RMS Pos':<18}"
    f"{'RMS Alt':<15}"
    f"{'RMS Heading':<18}"
)

print(
    "------------------------------------------------------------"
)

for result in all_results:

    print(
        f"{result['gnss_rate']:<12.1f}"
        f"{result['barometer_rate']:<12.1f}"
        f"{result['true_landing_error']:<18.3f}"
        f"{result['rms_position_error']:<18.3f}"
        f"{result['rms_altitude_error']:<15.3f}"
        f"{result['rms_heading_error']:<18.3f}"
    )

print(
    "============================================================"
)


# ============================================================
# BEST EKF CONFIGURATION
# ============================================================

print()
print("========================================")
print("BEST EKF ESTIMATION CONFIGURATION")
print("========================================")

print(
    "GNSS update rate:",
    best_estimator["gnss_rate"],
    "Hz"
)

print(
    "Barometer update rate:",
    best_estimator["barometer_rate"],
    "Hz"
)

print(
    "RMS position error:",
    best_estimator["rms_position_error"],
    "m"
)

print(
    "RMS altitude error:",
    best_estimator["rms_altitude_error"],
    "m"
)

print(
    "RMS heading error:",
    best_estimator["rms_heading_error"],
    "deg"
)


# ============================================================
# BEST LANDING CONFIGURATION
# ============================================================

print()
print("========================================")
print("BEST LANDING CONFIGURATION")
print("========================================")

print(
    "GNSS update rate:",
    best_landing["gnss_rate"],
    "Hz"
)

print(
    "Barometer update rate:",
    best_landing["barometer_rate"],
    "Hz"
)

print(
    "True landing position:",
    best_landing["true_x"],
    ",",
    best_landing["true_y"],
    "m"
)

print(
    "True landing error:",
    best_landing[
        "true_landing_error"
    ],
    "m"
)


# ============================================================
# V10.1 OVERALL ASSESSMENT
# ============================================================

print()
print("========================================")
print("V10.1 NAVIGATION SYSTEM ASSESSMENT")
print("========================================")

raw_gnss_error = (
    gnss_position_noise
)

print(
    "Raw GNSS position noise:",
    raw_gnss_error,
    "m"
)

print(
    "Best EKF RMS position error:",
    best_estimator[
        "rms_position_error"
    ],
    "m"
)

if (
    best_estimator[
        "rms_position_error"
    ]
    <
    raw_gnss_error
):

    print(
        "EKF STATUS: ESTIMATION IMPROVED"
    )

else:

    print(
        "EKF STATUS: FURTHER TUNING REQUIRED"
    )


# ============================================================
# PLOT 1
# LANDING ERROR VS GNSS RATE
# ============================================================

plt.figure()

for barometer_rate in barometer_rates:

    values = []

    for gnss_rate in gnss_rates:

        matching = [
            r for r in all_results
            if (
                r["gnss_rate"] ==
                gnss_rate
                and
                r["barometer_rate"] ==
                barometer_rate
            )
        ]

        values.append(
            matching[0][
                "true_landing_error"
            ]
        )

    plt.plot(
        gnss_rates,
        values,
        marker="o",
        label=(
            f"Barometer "
            f"{barometer_rate} Hz"
        )
    )

plt.xlabel(
    "GNSS Update Rate (Hz)"
)

plt.ylabel(
    "True Landing Error (m)"
)

plt.title(
    "V10.1 Landing Error vs GNSS Update Rate"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 2
# EKF RMS POSITION ERROR
# ============================================================

plt.figure()

for barometer_rate in barometer_rates:

    values = []

    for gnss_rate in gnss_rates:

        matching = [
            r for r in all_results
            if (
                r["gnss_rate"] ==
                gnss_rate
                and
                r["barometer_rate"] ==
                barometer_rate
            )
        ]

        values.append(
            matching[0][
                "rms_position_error"
            ]
        )

    plt.plot(
        gnss_rates,
        values,
        marker="o",
        label=(
            f"Barometer "
            f"{barometer_rate} Hz"
        )
    )

plt.xlabel(
    "GNSS Update Rate (Hz)"
)

plt.ylabel(
    "EKF RMS Position Error (m)"
)

plt.title(
    "V10.1 EKF Position Estimation vs GNSS Rate"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 3
# BEST CONFIGURATION TRAJECTORY
# ============================================================

plt.figure()

plt.plot(
    best_estimator[
        "true_x_history"
    ],
    best_estimator[
        "true_y_history"
    ],
    label="True trajectory"
)

plt.plot(
    best_estimator[
        "estimated_x_history"
    ],
    best_estimator[
        "estimated_y_history"
    ],
    linestyle="--",
    label="EKF trajectory"
)

plt.scatter(
    [target_x],
    [target_y],
    marker="x",
    s=100,
    label="Target"
)

plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "V10.1 True vs EKF Estimated Trajectory"
)

plt.grid()

plt.axis("equal")

plt.legend()

plt.show()


# ============================================================
# PLOT 4
# POSITION ESTIMATION ERROR
# ============================================================

plt.figure()

time_history = np.arange(
    len(
        best_estimator[
            "position_errors"
        ]
    )
) * dt

plt.plot(
    time_history,
    best_estimator[
        "position_errors"
    ]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Position Estimation Error (m)"
)

plt.title(
    "V10.1 EKF Position Estimation Error"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 5
# ALTITUDE ESTIMATION
# ============================================================

plt.figure()

plt.plot(
    best_estimator[
        "true_altitude_history"
    ],
    label="True altitude"
)

plt.plot(
    best_estimator[
        "estimated_altitude_history"
    ],
    linestyle="--",
    label="EKF altitude"
)

plt.xlabel(
    "Simulation Step"
)

plt.ylabel(
    "Altitude (m)"
)

plt.title(
    "V10.1 True vs EKF Altitude"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("V10.1 SIMULATION COMPLETE")
print("========================================")

print(
    "Reference wind:",
    wind_speed,
    "m/s"
)

print(
    "Reference direction:",
    wind_direction_deg,
    "degrees"
)

print(
    "GNSS rates tested:",
    gnss_rates,
    "Hz"
)

print(
    "Barometer rates tested:",
    barometer_rates,
    "Hz"
)

print(
    "IMU rate:",
    imu_update_rate,
    "Hz"
)

print(
    "Best EKF GNSS rate:",
    best_estimator[
        "gnss_rate"
    ],
    "Hz"
)

print(
    "Best EKF barometer rate:",
    best_estimator[
        "barometer_rate"
    ],
    "Hz"
)

print(
    "Best EKF RMS position error:",
    best_estimator[
        "rms_position_error"
    ],
    "m"
)

print(
    "Best landing error:",
    best_landing[
        "true_landing_error"
    ],
    "m"
)

print()
print(
    "NEXT DEVELOPMENT STEP:"
)

print(
    "V10.2 -> GNSS dropout + outlier rejection"
)

print("========================================")