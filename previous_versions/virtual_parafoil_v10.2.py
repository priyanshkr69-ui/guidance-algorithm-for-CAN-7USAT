import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V10.2
# GNSS DROPOUT + OUTLIER REJECTION
# ============================================================

np.random.seed(42)


print()
print("========================================")
print("VIRTUAL PARAFOIL V10.2")
print("GNSS DROPOUT + OUTLIER REJECTION")
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


# ============================================================
# TARGET
# ============================================================

target_x = 500.0
target_y = 200.0

reachability_tolerance = 20.0


# ============================================================
# AERODYNAMICS
# ============================================================

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
    airspeed *
    np.cos(glide_angle)
)

vertical_velocity = (
    airspeed *
    np.sin(glide_angle)
)

glide_ratio = (
    horizontal_air_velocity /
    vertical_velocity
)


# ============================================================
# REFERENCE WIND
# ============================================================

wind_speed = 3.0
wind_direction_deg = 0.0

wind_direction = np.radians(
    wind_direction_deg
)

wind_x = (
    wind_speed *
    np.cos(wind_direction)
)

wind_y = (
    wind_speed *
    np.sin(wind_direction)
)


# ============================================================
# SENSOR MODEL
# ============================================================

gnss_position_noise = 3.0
gnss_velocity_noise = 0.3

barometer_altitude_noise = 2.0

imu_heading_noise_deg = 2.0
imu_heading_noise = np.radians(
    imu_heading_noise_deg
)

imu_turn_rate_noise_deg = 0.5
imu_turn_rate_noise = np.radians(
    imu_turn_rate_noise_deg
)


# ============================================================
# SENSOR UPDATE RATES
# ============================================================

gnss_rate = 5.0
barometer_rate = 10.0
imu_rate = 50.0


# ============================================================
# GNSS FAILURE SETTINGS
# ============================================================

# Set True to simulate GNSS dropout
ENABLE_GNSS_DROPOUT = True

# Set True to simulate GNSS outliers
ENABLE_GNSS_OUTLIERS = True

# Fraction of GNSS measurements affected by dropout
GNSS_DROPOUT_PROBABILITY = 0.10

# Probability that a valid GNSS measurement becomes an outlier
GNSS_OUTLIER_PROBABILITY = 0.05

# Outlier magnitude
GNSS_POSITION_OUTLIER_SIZE = 40.0
GNSS_VELOCITY_OUTLIER_SIZE = 5.0


# ============================================================
# GUIDANCE PARAMETERS
# ============================================================

max_turn_rate = np.radians(15.0)

guidance_interval = 2.0

candidate_commands = np.linspace(
    -1.0,
    1.0,
    21
)


# ============================================================
# ADAPTIVE HORIZON
# ============================================================

def get_prediction_horizon(
    altitude
):

    if altitude > 400.0:
        return 20.0

    elif altitude > 200.0:
        return 15.0

    elif altitude > 100.0:
        return 10.0

    else:
        return 5.0


# ============================================================
# PRINT PARAMETERS
# ============================================================

print()
print("========================================")
print("PARAFOIL PARAMETERS")
print("========================================")

print(
    "Area:",
    area,
    "m^2"
)

print(
    "Mass:",
    mass,
    "kg"
)

print(
    "CL:",
    CL
)

print(
    "CD:",
    CD
)

print(
    "Airspeed:",
    airspeed,
    "m/s"
)

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

print(
    "Glide ratio:",
    glide_ratio
)


# ============================================================
# SENSOR PRINT
# ============================================================

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
    imu_heading_noise_deg,
    "deg"
)

print(
    "IMU turn-rate noise:",
    imu_turn_rate_noise_deg,
    "deg/s"
)

print(
    "GNSS update rate:",
    gnss_rate,
    "Hz"
)

print(
    "Barometer update rate:",
    barometer_rate,
    "Hz"
)

print(
    "IMU update rate:",
    imu_rate,
    "Hz"
)


# ============================================================
# FAILURE PRINT
# ============================================================

print()
print("========================================")
print("GNSS FAILURE MODEL")
print("========================================")

print(
    "GNSS dropout enabled:",
    ENABLE_GNSS_DROPOUT
)

print(
    "GNSS dropout probability:",
    GNSS_DROPOUT_PROBABILITY
)

print(
    "GNSS outlier rejection enabled:",
    ENABLE_GNSS_OUTLIERS
)

print(
    "GNSS outlier probability:",
    GNSS_OUTLIER_PROBABILITY
)

print(
    "Position outlier magnitude:",
    GNSS_POSITION_OUTLIER_SIZE,
    "m"
)

print(
    "Velocity outlier magnitude:",
    GNSS_VELOCITY_OUTLIER_SIZE,
    "m/s"
)


# ============================================================
# TARGET PRINT
# ============================================================

print()
print("========================================")
print("TARGET")
print("========================================")

print(
    "Target X:",
    target_x,
    "m"
)

print(
    "Target Y:",
    target_y,
    "m"
)

print(
    "Tolerance:",
    reachability_tolerance,
    "m"
)


# ============================================================
# WIND PRINT
# ============================================================

print()
print("========================================")
print("REFERENCE WIND")
print("========================================")

print(
    "Wind speed:",
    wind_speed,
    "m/s"
)

print(
    "Wind direction:",
    wind_direction_deg,
    "degrees"
)

print(
    "Wind X:",
    wind_x,
    "m/s"
)

print(
    "Wind Y:",
    wind_y,
    "m/s"
)


# ============================================================
# EKF CLASS
# ============================================================

class ParafoilEKF:

    def __init__(self):

        # State:
        #
        # x
        # y
        # vx
        # vy
        # altitude
        # heading

        self.x = np.array(
            [
                0.0,
                0.0,
                horizontal_air_velocity,
                0.0,
                600.0,
                0.0
            ],
            dtype=float
        )

        # Covariance

        self.P = np.diag(
            [
                25.0,
                25.0,
                4.0,
                4.0,
                16.0,
                np.radians(10.0) ** 2
            ]
        )

        # Process noise

        self.Q = np.diag(
            [
                0.05,
                0.05,
                0.20,
                0.20,
                0.10,
                np.radians(0.5) ** 2
            ]
        )

        # Measurement noise

        self.R_gnss = np.diag(
            [
                gnss_position_noise ** 2,
                gnss_position_noise ** 2,
                gnss_velocity_noise ** 2,
                gnss_velocity_noise ** 2
            ]
        )

        self.R_baro = np.array(
            [
                [barometer_altitude_noise ** 2]
            ]
        )

        self.R_heading = np.array(
            [
                [imu_heading_noise ** 2]
            ]
        )

        self.accepted_gnss = 0
        self.rejected_gnss = 0


    # ========================================================
    # ANGLE NORMALIZATION
    # ========================================================

    def normalize_angle(
        self,
        angle
    ):

        return (
            angle + np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        dt,
        steering_command
    ):

        heading = self.x[5]

        turn_rate = (
            max_turn_rate *
            steering_command
        )

        new_heading = (
            heading +
            turn_rate *
            dt
        )

        new_heading = self.normalize_angle(
            new_heading
        )

        vx_air = (
            horizontal_air_velocity *
            np.cos(new_heading)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(new_heading)
        )

        vx_ground = (
            vx_air +
            wind_x
        )

        vy_ground = (
            vy_air +
            wind_y
        )

        self.x[0] += (
            vx_ground *
            dt
        )

        self.x[1] += (
            vy_ground *
            dt
        )

        self.x[2] = vx_ground
        self.x[3] = vy_ground

        self.x[4] -= (
            vertical_velocity *
            dt
        )

        self.x[5] = new_heading


        # ----------------------------------------------------
        # Numerical state transition matrix
        # ----------------------------------------------------

        F = np.eye(6)

        F[0, 2] = dt
        F[1, 3] = dt

        F[5, 5] = 1.0

        self.P = (
            F @
            self.P @
            F.T
            +
            self.Q
        )


    # ========================================================
    # GNSS UPDATE WITH INNOVATION REJECTION
    # ========================================================

    def update_gnss(
        self,
        measurement
    ):

        z = np.asarray(
            measurement
        )

        H = np.zeros(
            (4, 6)
        )

        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0

        predicted = (
            H @
            self.x
        )

        innovation = (
            z -
            predicted
        )

        S = (
            H @
            self.P @
            H.T
            +
            self.R_gnss
        )

        try:

            mahalanobis = (
                innovation.T @
                np.linalg.inv(S) @
                innovation
            )

        except np.linalg.LinAlgError:

            mahalanobis = 999.0


        # ----------------------------------------------------
        # Outlier rejection threshold
        #
        # 4-dimensional measurement
        # Conservative threshold
        # ----------------------------------------------------

        threshold = 13.28


        if mahalanobis > threshold:

            self.rejected_gnss += 1

            return False


        # ----------------------------------------------------
        # Kalman update
        # ----------------------------------------------------

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x = (
            self.x +
            K @ innovation
        )

        I = np.eye(6)

        self.P = (
            I -
            K @ H
        ) @ self.P

        self.accepted_gnss += 1

        return True


    # ========================================================
    # BAROMETER UPDATE
    # ========================================================

    def update_barometer(
        self,
        altitude_measurement
    ):

        H = np.zeros(
            (1, 6)
        )

        H[0, 4] = 1.0

        z = np.array(
            [
                altitude_measurement
            ]
        )

        predicted = (
            H @
            self.x
        )

        innovation = (
            z -
            predicted
        )

        S = (
            H @
            self.P @
            H.T
            +
            self.R_baro
        )

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x = (
            self.x +
            K @ innovation
        )

        I = np.eye(6)

        self.P = (
            I -
            K @ H
        ) @ self.P


    # ========================================================
    # HEADING UPDATE
    # ========================================================

    def update_heading(
        self,
        heading_measurement
    ):

        H = np.zeros(
            (1, 6)
        )

        H[0, 5] = 1.0

        innovation = (
            heading_measurement -
            self.x[5]
        )

        innovation = self.normalize_angle(
            innovation
        )

        S = (
            H @
            self.P @
            H.T
            +
            self.R_heading
        )

        K = (
            self.P @
            H.T @
            np.linalg.inv(S)
        )

        self.x = (
            self.x +
            K[:, 0] *
            innovation
        )

        self.x[5] = self.normalize_angle(
            self.x[5]
        )

        I = np.eye(6)

        self.P = (
            I -
            K @ H
        ) @ self.P


# ============================================================
# GUIDANCE FUNCTION
# ============================================================

def select_guidance_command(
    x,
    y,
    altitude,
    heading
):

    horizon = get_prediction_horizon(
        altitude
    )

    best_command = 0.0

    best_cost = float(
        "inf"
    )


    for command in candidate_commands:

        px = x
        py = y
        pheading = heading

        prediction_dt = 0.5

        steps = max(
            1,
            int(
                horizon /
                prediction_dt
            )
        )

        for _ in range(steps):

            turn_rate = (
                max_turn_rate *
                command
            )

            pheading += (
                turn_rate *
                prediction_dt
            )

            pheading = (
                pheading +
                np.pi
            ) % (
                2.0 * np.pi
            ) - np.pi


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
                wind_x
            )

            vy_ground = (
                vy_air +
                wind_y
            )

            px += (
                vx_ground *
                prediction_dt
            )

            py += (
                vy_ground *
                prediction_dt
            )


        position_error = np.sqrt(
            (
                px -
                target_x
            ) ** 2
            +
            (
                py -
                target_y
            ) ** 2
        )


        desired_heading = np.arctan2(
            target_y - y,
            target_x - x
        )

        heading_error = (
            desired_heading -
            heading
        )

        heading_error = (
            heading_error +
            np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi


        heading_cost = abs(
            heading_error
        )


        # ----------------------------------------------------
        # Cost function
        # ----------------------------------------------------

        position_weight = 0.75
        heading_weight = 0.25

        cost = (
            position_weight *
            position_error
            +
            heading_weight *
            30.0 *
            heading_cost
        )


        if cost < best_cost:

            best_cost = cost
            best_command = command


    return best_command


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation():

    # --------------------------------------------------------
    # TRUE STATE
    # --------------------------------------------------------

    altitude = 600.0

    true_x = 0.0
    true_y = 0.0

    true_heading = 0.0

    dt = 1.0 / imu_rate

    time = 0.0

    next_guidance_update = 0.0

    next_gnss_update = 0.0

    next_baro_update = 0.0

    current_steering = 0.0


    # --------------------------------------------------------
    # EKF
    # --------------------------------------------------------

    ekf = ParafoilEKF()


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    true_x_history = []
    true_y_history = []

    estimated_x_history = []
    estimated_y_history = []

    true_altitude_history = []
    estimated_altitude_history = []

    true_heading_history = []
    estimated_heading_history = []

    steering_history = []

    time_history = []

    gnss_available_history = []
    gnss_accepted_history = []


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    position_errors = []
    altitude_errors = []
    heading_errors = []


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while altitude > 0.0:

        # ----------------------------------------------------
        # GUIDANCE
        # ----------------------------------------------------

        if time >= next_guidance_update:

            current_steering = (
                select_guidance_command(
                    ekf.x[0],
                    ekf.x[1],
                    max(
                        ekf.x[4],
                        0.0
                    ),
                    ekf.x[5]
                )
            )

            next_guidance_update = (
                time +
                guidance_interval
            )


        # ----------------------------------------------------
        # TRUE PARAFOIL DYNAMICS
        # ----------------------------------------------------

        true_turn_rate = (
            max_turn_rate *
            current_steering
        )

        true_heading += (
            true_turn_rate *
            dt
        )

        true_heading = (
            true_heading +
            np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi


        vx_air = (
            horizontal_air_velocity *
            np.cos(true_heading)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(true_heading)
        )

        vx_ground = (
            vx_air +
            wind_x
        )

        vy_ground = (
            vy_air +
            wind_y
        )


        true_x += (
            vx_ground *
            dt
        )

        true_y += (
            vy_ground *
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
            dt,
            current_steering
        )


        # ----------------------------------------------------
        # GNSS UPDATE
        # ----------------------------------------------------

        gnss_available = False
        gnss_accepted = False


        if time >= next_gnss_update:

            next_gnss_update = (
                time +
                1.0 /
                gnss_rate
            )


            # Random dropout

            dropout = False

            if ENABLE_GNSS_DROPOUT:

                if (
                    np.random.rand()
                    <
                    GNSS_DROPOUT_PROBABILITY
                ):

                    dropout = True


            if not dropout:

                gnss_available = True


                measured_x = (
                    true_x +
                    np.random.normal(
                        0.0,
                        gnss_position_noise
                    )
                )

                measured_y = (
                    true_y +
                    np.random.normal(
                        0.0,
                        gnss_position_noise
                    )
                )

                measured_vx = (
                    vx_ground +
                    np.random.normal(
                        0.0,
                        gnss_velocity_noise
                    )
                )

                measured_vy = (
                    vy_ground +
                    np.random.normal(
                        0.0,
                        gnss_velocity_noise
                    )
                )


                # ------------------------------------------------
                # Inject outlier
                # ------------------------------------------------

                outlier = False

                if ENABLE_GNSS_OUTLIERS:

                    if (
                        np.random.rand()
                        <
                        GNSS_OUTLIER_PROBABILITY
                    ):

                        outlier = True


                if outlier:

                    measured_x += (
                        np.random.choice(
                            [-1.0, 1.0]
                        )
                        *
                        GNSS_POSITION_OUTLIER_SIZE
                    )

                    measured_y += (
                        np.random.choice(
                            [-1.0, 1.0]
                        )
                        *
                        GNSS_POSITION_OUTLIER_SIZE
                    )

                    measured_vx += (
                        np.random.choice(
                            [-1.0, 1.0]
                        )
                        *
                        GNSS_VELOCITY_OUTLIER_SIZE
                    )

                    measured_vy += (
                        np.random.choice(
                            [-1.0, 1.0]
                        )
                        *
                        GNSS_VELOCITY_OUTLIER_SIZE
                    )


                gnss_accepted = (
                    ekf.update_gnss(
                        [
                            measured_x,
                            measured_y,
                            measured_vx,
                            measured_vy
                        ]
                    )
                )


        # ----------------------------------------------------
        # BAROMETER
        # ----------------------------------------------------

        if time >= next_baro_update:

            next_baro_update = (
                time +
                1.0 /
                barometer_rate
            )

            measured_altitude = (
                altitude +
                np.random.normal(
                    0.0,
                    barometer_altitude_noise
                )
            )

            ekf.update_barometer(
                measured_altitude
            )


        # ----------------------------------------------------
        # IMU HEADING UPDATE
        # ----------------------------------------------------

        measured_heading = (
            true_heading +
            np.random.normal(
                0.0,
                imu_heading_noise
            )
        )

        measured_heading = (
            measured_heading +
            np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi


        ekf.update_heading(
            measured_heading
        )


        # ----------------------------------------------------
        # ERRORS
        # ----------------------------------------------------

        position_error = np.sqrt(
            (
                true_x -
                ekf.x[0]
            ) ** 2
            +
            (
                true_y -
                ekf.x[1]
            ) ** 2
        )

        altitude_error = abs(
            altitude -
            ekf.x[4]
        )

        heading_error = (
            true_heading -
            ekf.x[5]
        )

        heading_error = (
            heading_error +
            np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi


        position_errors.append(
            position_error
        )

        altitude_errors.append(
            altitude_error
        )

        heading_errors.append(
            np.degrees(
                abs(
                    heading_error
                )
            )
        )


        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        true_x_history.append(
            true_x
        )

        true_y_history.append(
            true_y
        )

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
            ekf.x[4]
        )

        true_heading_history.append(
            np.degrees(
                true_heading
            )
        )

        estimated_heading_history.append(
            np.degrees(
                ekf.x[5]
            )
        )

        steering_history.append(
            current_steering
        )

        time_history.append(
            time
        )

        gnss_available_history.append(
            gnss_available
        )

        gnss_accepted_history.append(
            gnss_accepted
        )


        time += dt


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    true_landing_error = np.sqrt(
        (
            true_x -
            target_x
        ) ** 2
        +
        (
            true_y -
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


    # --------------------------------------------------------
    # Steering statistics
    # --------------------------------------------------------

    steering_array = np.array(
        steering_history
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


    # --------------------------------------------------------
    # EKF statistics
    # --------------------------------------------------------

    position_errors = np.array(
        position_errors
    )

    altitude_errors = np.array(
        altitude_errors
    )

    heading_errors = np.array(
        heading_errors
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

    max_altitude_error = np.max(
        altitude_errors
    )


    mean_heading_error = np.mean(
        heading_errors
    )

    rms_heading_error = np.sqrt(
        np.mean(
            heading_errors ** 2
        )
    )

    max_heading_error = np.max(
        heading_errors
    )


    # --------------------------------------------------------
    # GNSS statistics
    # --------------------------------------------------------

    total_gnss_updates = np.sum(
        gnss_available_history
    )

    accepted_gnss_updates = (
        ekf.accepted_gnss
    )

    rejected_gnss_updates = (
        ekf.rejected_gnss
    )

    dropout_count = (
        len(gnss_available_history)
        -
        total_gnss_updates
    )


    if total_gnss_updates > 0:

        rejection_percentage = (
            rejected_gnss_updates /
            total_gnss_updates
            *
            100.0
        )

    else:

        rejection_percentage = 0.0


    return {

        "true_x": true_x,

        "true_y": true_y,

        "estimated_x": ekf.x[0],

        "estimated_y": ekf.x[1],

        "true_landing_error":
            true_landing_error,

        "estimated_landing_error":
            estimated_landing_error,

        "flight_time":
            time,

        "average_steering":
            average_steering,

        "steering_reversals":
            steering_reversals,

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

        "total_gnss_updates":
            total_gnss_updates,

        "accepted_gnss_updates":
            accepted_gnss_updates,

        "rejected_gnss_updates":
            rejected_gnss_updates,

        "dropout_count":
            dropout_count,

        "rejection_percentage":
            rejection_percentage,

        "time":
            time_history,

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

        "steering_history":
            steering_history,

        "gnss_available_history":
            gnss_available_history,

        "gnss_accepted_history":
            gnss_accepted_history
    }


# ============================================================
# GUIDANCE PRINT
# ============================================================

print()
print("========================================")
print("GUIDANCE")
print("========================================")

print(
    "Guidance interval:",
    guidance_interval,
    "s"
)

print(
    "Candidate commands:",
    len(candidate_commands)
)

print(
    "Maximum turn rate:",
    np.degrees(
        max_turn_rate
    ),
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


# ============================================================
# RUN
# ============================================================

print()
print("========================================")
print("STARTING V10.2 SIMULATION")
print("========================================")


simulation = run_simulation()


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("========================================")
print("V10.2 FINAL RESULTS")
print("========================================")

print(
    "True landing X:",
    simulation["true_x"],
    "m"
)

print(
    "True landing Y:",
    simulation["true_y"],
    "m"
)

print(
    "Estimated landing X:",
    simulation["estimated_x"],
    "m"
)

print(
    "Estimated landing Y:",
    simulation["estimated_y"],
    "m"
)

print(
    "Target X:",
    target_x,
    "m"
)

print(
    "Target Y:",
    target_y,
    "m"
)

print(
    "True landing error:",
    simulation[
        "true_landing_error"
    ],
    "m"
)

print(
    "Estimated landing error:",
    simulation[
        "estimated_landing_error"
    ],
    "m"
)

print(
    "Flight time:",
    simulation[
        "flight_time"
    ],
    "s"
)

print(
    "Average steering:",
    simulation[
        "average_steering"
    ]
)

print(
    "Steering reversals:",
    simulation[
        "steering_reversals"
    ]
)


# ============================================================
# EKF PERFORMANCE
# ============================================================

print()
print("========================================")
print("EKF STATE ESTIMATION PERFORMANCE")
print("========================================")

print(
    "Mean position estimation error:",
    simulation[
        "mean_position_error"
    ],
    "m"
)

print(
    "RMS position estimation error:",
    simulation[
        "rms_position_error"
    ],
    "m"
)

print(
    "Maximum position estimation error:",
    simulation[
        "max_position_error"
    ],
    "m"
)

print(
    "Mean altitude estimation error:",
    simulation[
        "mean_altitude_error"
    ],
    "m"
)

print(
    "RMS altitude estimation error:",
    simulation[
        "rms_altitude_error"
    ],
    "m"
)

print(
    "Maximum altitude estimation error:",
    simulation[
        "max_altitude_error"
    ],
    "m"
)

print(
    "Mean heading estimation error:",
    simulation[
        "mean_heading_error"
    ],
    "degrees"
)

print(
    "RMS heading estimation error:",
    simulation[
        "rms_heading_error"
    ],
    "degrees"
)

print(
    "Maximum heading estimation error:",
    simulation[
        "max_heading_error"
    ],
    "degrees"
)


# ============================================================
# GNSS FAILURE PERFORMANCE
# ============================================================

print()
print("========================================")
print("GNSS FAILURE / OUTLIER PERFORMANCE")
print("========================================")

print(
    "GNSS measurements available:",
    simulation[
        "total_gnss_updates"
    ]
)

print(
    "GNSS measurements accepted:",
    simulation[
        "accepted_gnss_updates"
    ]
)

print(
    "GNSS measurements rejected:",
    simulation[
        "rejected_gnss_updates"
    ]
)

print(
    "GNSS dropout events:",
    simulation[
        "dropout_count"
    ]
)

print(
    "GNSS rejection percentage:",
    simulation[
        "rejection_percentage"
    ],
    "%"
)


# ============================================================
# NAVIGATION ASSESSMENT
# ============================================================

print()
print("========================================")
print("V10.2 NAVIGATION SYSTEM ASSESSMENT")
print("========================================")

print(
    "Raw GNSS position noise:",
    gnss_position_noise,
    "m"
)

print(
    "EKF RMS position error:",
    simulation[
        "rms_position_error"
    ],
    "m"
)


if (
    simulation[
        "rms_position_error"
    ]
    <
    gnss_position_noise
):

    print(
        "EKF STATUS: ESTIMATION IMPROVED"
    )

else:

    print(
        "EKF STATUS: NEEDS FURTHER TUNING"
    )


if (
    simulation[
        "rejected_gnss_updates"
    ]
    > 0
):

    print(
        "OUTLIER REJECTION: ACTIVE"
    )

else:

    print(
        "OUTLIER REJECTION: NO OUTLIERS REJECTED"
    )


# ============================================================
# LANDING ASSESSMENT
# ============================================================

print()
print("========================================")
print("LANDING PERFORMANCE")
print("========================================")

if (
    simulation[
        "true_landing_error"
    ]
    <=
    reachability_tolerance
):

    print(
        "LANDING STATUS: WITHIN TOLERANCE"
    )

else:

    print(
        "LANDING STATUS: OUTSIDE TOLERANCE"
    )


# ============================================================
# PLOT 1: TRUE VS ESTIMATED TRAJECTORY
# ============================================================

plt.figure()

plt.plot(
    simulation[
        "true_x_history"
    ],
    simulation[
        "true_y_history"
    ],
    label="True trajectory"
)

plt.plot(
    simulation[
        "estimated_x_history"
    ],
    simulation[
        "estimated_y_history"
    ],
    linestyle="--",
    label="EKF trajectory"
)

plt.scatter(
    target_x,
    target_y,
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
    "V10.2 True vs EKF Estimated Trajectory"
)

plt.grid()

plt.legend()

plt.axis(
    "equal"
)

plt.show()


# ============================================================
# PLOT 2: POSITION ESTIMATION ERROR
# ============================================================

position_error_history = []

for tx, ty, ex, ey in zip(
    simulation[
        "true_x_history"
    ],
    simulation[
        "true_y_history"
    ],
    simulation[
        "estimated_x_history"
    ],
    simulation[
        "estimated_y_history"
    ]
):

    position_error_history.append(
        np.sqrt(
            (
                tx -
                ex
            ) ** 2
            +
            (
                ty -
                ey
            ) ** 2
        )
    )


plt.figure()

plt.plot(
    simulation["time"],
    position_error_history
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Position Estimation Error (m)"
)

plt.title(
    "V10.2 EKF Position Estimation Error"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 3: ALTITUDE ESTIMATION
# ============================================================

plt.figure()

plt.plot(
    simulation["time"],
    simulation[
        "true_altitude_history"
    ],
    label="True altitude"
)

plt.plot(
    simulation["time"],
    simulation[
        "estimated_altitude_history"
    ],
    linestyle="--",
    label="EKF altitude"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Altitude (m)"
)

plt.title(
    "V10.2 True vs Estimated Altitude"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 4: HEADING
# ============================================================

plt.figure()

plt.plot(
    simulation["time"],
    simulation[
        "true_heading_history"
    ],
    label="True heading"
)

plt.plot(
    simulation["time"],
    simulation[
        "estimated_heading_history"
    ],
    linestyle="--",
    label="EKF heading"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Heading (deg)"
)

plt.title(
    "V10.2 True vs Estimated Heading"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 5: STEERING
# ============================================================

plt.figure()

plt.plot(
    simulation["time"],
    simulation[
        "steering_history"
    ]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Steering Command"
)

plt.title(
    "V10.2 Steering Command"
)

plt.grid()

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("V10.2 SIMULATION COMPLETE")
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
    "GNSS rate:",
    gnss_rate,
    "Hz"
)

print(
    "Barometer rate:",
    barometer_rate,
    "Hz"
)

print(
    "IMU rate:",
    imu_rate,
    "Hz"
)

print(
    "GNSS dropout probability:",
    GNSS_DROPOUT_PROBABILITY
)

print(
    "GNSS outlier probability:",
    GNSS_OUTLIER_PROBABILITY
)

print(
    "EKF RMS position error:",
    simulation[
        "rms_position_error"
    ],
    "m"
)

print(
    "True landing error:",
    simulation[
        "true_landing_error"
    ],
    "m"
)

print(
    "GNSS dropouts:",
    simulation[
        "dropout_count"
    ]
)

print(
    "GNSS rejected outliers:",
    simulation[
        "rejected_gnss_updates"
    ]
)

print(
    "========================================"
)

print(
    "NEXT DEVELOPMENT STEP:"
)

print(
    "V10.3 -> EKF + GNSS failure-duration robustness study"
)

print(
    "========================================"
)