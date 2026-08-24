import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V10.0
# SENSOR NOISE + EKF STATE ESTIMATION
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V10.0")
print("SENSOR NOISE + EKF STATE ESTIMATION")
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
# CONTROL PARAMETERS
# ============================================================

max_turn_rate = np.radians(15.0)

guidance_interval = 2.0

candidate_commands = np.linspace(
    -1.0,
    1.0,
    21
)


# ============================================================
# ADAPTIVE GUIDANCE HORIZON
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

# GNSS position standard deviation
GNSS_POSITION_STD = 3.0       # m

# GNSS velocity standard deviation
GNSS_VELOCITY_STD = 0.30      # m/s

# Barometer altitude standard deviation
BARO_ALTITUDE_STD = 2.0       # m

# IMU heading standard deviation
IMU_HEADING_STD = np.radians(2.0)

# IMU turn-rate standard deviation
IMU_TURN_RATE_STD = np.radians(0.5)


# ============================================================
# AERODYNAMICS
# ============================================================

weight = mass * g

glide_angle = np.arctan(
    CD / CL
)

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
# EKF STATE
#
# STATE VECTOR:
#
# X =
# [ x
#   y
#   altitude
#   heading
#   vx
#   vy ]
# ============================================================

STATE_SIZE = 6


# ============================================================
# ANGLE NORMALIZATION
# ============================================================

def normalize_angle(angle):

    return (
        angle + np.pi
    ) % (
        2 * np.pi
    ) - np.pi


# ============================================================
# EKF INITIALIZATION
# ============================================================

def initialize_ekf(
    x0,
    y0,
    altitude0,
    heading0,
    vx0,
    vy0
):

    state = np.array([
        x0,
        y0,
        altitude0,
        heading0,
        vx0,
        vy0
    ], dtype=float)

    covariance = np.diag([
        10.0**2,
        10.0**2,
        5.0**2,
        np.radians(10.0)**2,
        1.0**2,
        1.0**2
    ])

    return state, covariance


# ============================================================
# EKF PREDICTION
# ============================================================

def ekf_predict(
    state,
    covariance,
    steering_command,
    wind_x,
    wind_y,
    dt
):

    x = state[0]
    y = state[1]
    altitude = state[2]
    heading = state[3]
    vx = state[4]
    vy = state[5]

    # --------------------------------------------------------
    # Parafoil turn dynamics
    # --------------------------------------------------------

    turn_rate = (
        max_turn_rate *
        steering_command
    )

    new_heading = (
        heading +
        turn_rate *
        dt
    )

    new_heading = normalize_angle(
        new_heading
    )

    # --------------------------------------------------------
    # Air-relative velocity
    # --------------------------------------------------------

    vx_air = (
        horizontal_air_velocity *
        np.cos(new_heading)
    )

    vy_air = (
        horizontal_air_velocity *
        np.sin(new_heading)
    )

    # --------------------------------------------------------
    # Ground velocity
    # --------------------------------------------------------

    new_vx = (
        vx_air +
        wind_x
    )

    new_vy = (
        vy_air +
        wind_y
    )

    # --------------------------------------------------------
    # State propagation
    # --------------------------------------------------------

    new_x = (
        x +
        new_vx *
        dt
    )

    new_y = (
        y +
        new_vy *
        dt
    )

    new_altitude = (
        altitude -
        vertical_velocity *
        dt
    )

    predicted_state = np.array([
        new_x,
        new_y,
        new_altitude,
        new_heading,
        new_vx,
        new_vy
    ])

    # --------------------------------------------------------
    # State transition matrix
    # --------------------------------------------------------

    F = np.eye(
        STATE_SIZE
    )

    F[0, 4] = dt
    F[1, 5] = dt

    # --------------------------------------------------------
    # Process noise
    # --------------------------------------------------------

    Q = np.diag([
        0.5**2,
        0.5**2,
        0.5**2,
        np.radians(1.0)**2,
        0.5**2,
        0.5**2
    ])

    covariance = (
        F @
        covariance @
        F.T
        +
        Q
    )

    predicted_state[3] = normalize_angle(
        predicted_state[3]
    )

    return (
        predicted_state,
        covariance
    )


# ============================================================
# GNSS MEASUREMENT UPDATE
#
# Measurements:
#
# x
# y
# vx
# vy
# ============================================================

def ekf_gnss_update(
    state,
    covariance,
    measured_x,
    measured_y,
    measured_vx,
    measured_vy
):

    z = np.array([
        measured_x,
        measured_y,
        measured_vx,
        measured_vy
    ])

    H = np.zeros(
        (4, STATE_SIZE)
    )

    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 4] = 1.0
    H[3, 5] = 1.0

    R = np.diag([
        GNSS_POSITION_STD**2,
        GNSS_POSITION_STD**2,
        GNSS_VELOCITY_STD**2,
        GNSS_VELOCITY_STD**2
    ])

    predicted_measurement = (
        H @ state
    )

    innovation = (
        z -
        predicted_measurement
    )

    S = (
        H @
        covariance @
        H.T
        +
        R
    )

    K = (
        covariance @
        H.T @
        np.linalg.inv(S)
    )

    state = (
        state +
        K @ innovation
    )

    I = np.eye(
        STATE_SIZE
    )

    covariance = (
        I -
        K @ H
    ) @ covariance

    state[3] = normalize_angle(
        state[3]
    )

    return (
        state,
        covariance
    )


# ============================================================
# BAROMETER UPDATE
# ============================================================

def ekf_baro_update(
    state,
    covariance,
    measured_altitude
):

    z = np.array([
        measured_altitude
    ])

    H = np.zeros(
        (1, STATE_SIZE)
    )

    H[0, 2] = 1.0

    R = np.array([
        [BARO_ALTITUDE_STD**2]
    ])

    predicted_measurement = (
        H @ state
    )

    innovation = (
        z -
        predicted_measurement
    )

    S = (
        H @
        covariance @
        H.T
        +
        R
    )

    K = (
        covariance @
        H.T @
        np.linalg.inv(S)
    )

    state = (
        state +
        (
            K @ innovation
        )
    )

    I = np.eye(
        STATE_SIZE
    )

    covariance = (
        I -
        K @ H
    ) @ covariance

    state[3] = normalize_angle(
        state[3]
    )

    return (
        state,
        covariance
    )


# ============================================================
# IMU HEADING UPDATE
# ============================================================

def ekf_imu_update(
    state,
    covariance,
    measured_heading
):

    innovation = normalize_angle(
        measured_heading -
        state[3]
    )

    H = np.zeros(
        (1, STATE_SIZE)
    )

    H[0, 3] = 1.0

    R = np.array([
        [IMU_HEADING_STD**2]
    ])

    S = (
        H @
        covariance @
        H.T
        +
        R
    )

    K = (
        covariance @
        H.T @
        np.linalg.inv(S)
    )

    state = (
        state +
        (
            K.flatten() *
            innovation
        )
    )

    I = np.eye(
        STATE_SIZE
    )

    covariance = (
        I -
        K @ H
    ) @ covariance

    state[3] = normalize_angle(
        state[3]
    )

    return (
        state,
        covariance
    )


# ============================================================
# SENSOR SIMULATION
# ============================================================

def generate_sensor_measurements(
    true_state,
    true_turn_rate
):

    true_x = true_state[0]
    true_y = true_state[1]
    true_altitude = true_state[2]
    true_heading = true_state[3]
    true_vx = true_state[4]
    true_vy = true_state[5]

    # --------------------------------------------------------
    # GNSS
    # --------------------------------------------------------

    measured_x = (
        true_x +
        np.random.normal(
            0.0,
            GNSS_POSITION_STD
        )
    )

    measured_y = (
        true_y +
        np.random.normal(
            0.0,
            GNSS_POSITION_STD
        )
    )

    measured_vx = (
        true_vx +
        np.random.normal(
            0.0,
            GNSS_VELOCITY_STD
        )
    )

    measured_vy = (
        true_vy +
        np.random.normal(
            0.0,
            GNSS_VELOCITY_STD
        )
    )

    # --------------------------------------------------------
    # Barometer
    # --------------------------------------------------------

    measured_altitude = (
        true_altitude +
        np.random.normal(
            0.0,
            BARO_ALTITUDE_STD
        )
    )

    # --------------------------------------------------------
    # IMU
    # --------------------------------------------------------

    measured_heading = normalize_angle(
        true_heading +
        np.random.normal(
            0.0,
            IMU_HEADING_STD
        )
    )

    measured_turn_rate = (
        true_turn_rate +
        np.random.normal(
            0.0,
            IMU_TURN_RATE_STD
        )
    )

    return (
        measured_x,
        measured_y,
        measured_altitude,
        measured_heading,
        measured_vx,
        measured_vy,
        measured_turn_rate
    )


# ============================================================
# PREDICT LANDING USING ESTIMATED STATE
# ============================================================

def predict_landing(
    current_x,
    current_y,
    current_altitude,
    current_heading,
    steering_command,
    wind_x,
    wind_y,
    prediction_horizon
):

    remaining_time = (
        current_altitude /
        vertical_velocity
    )

    prediction_time = min(
        prediction_horizon,
        remaining_time
    )

    prediction_dt = 0.5

    steps = max(
        1,
        int(
            prediction_time /
            prediction_dt
        )
    )

    px = current_x
    py = current_y
    pheading = current_heading

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

    # --------------------------------------------------------
    # Continue straight after horizon
    # --------------------------------------------------------

    remaining_prediction_time = (
        remaining_time -
        prediction_time
    )

    if remaining_prediction_time > 0:

        vx_air = (
            horizontal_air_velocity *
            np.cos(pheading)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(pheading)
        )

        px += (
            (
                vx_air +
                wind_x
            ) *
            remaining_prediction_time
        )

        py += (
            (
                vy_air +
                wind_y
            ) *
            remaining_prediction_time
        )

    return (
        px,
        py
    )


# ============================================================
# GUIDANCE COMMAND SELECTION
# ============================================================

def select_guidance_command(
    estimated_state,
    wind_x,
    wind_y
):

    x = estimated_state[0]
    y = estimated_state[1]
    altitude = estimated_state[2]
    heading = estimated_state[3]

    prediction_horizon = (
        get_prediction_horizon(
            altitude
        )
    )

    best_command = 0.0
    best_error = float("inf")

    for command in candidate_commands:

        predicted_x, predicted_y = (
            predict_landing(
                x,
                y,
                altitude,
                heading,
                command,
                wind_x,
                wind_y,
                prediction_horizon
            )
        )

        error = np.sqrt(
            (
                predicted_x -
                target_x
            )**2
            +
            (
                predicted_y -
                target_y
            )**2
        )

        if error < best_error:

            best_error = error
            best_command = command

    return (
        best_command,
        best_error,
        prediction_horizon
    )


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation(
    wind_x,
    wind_y
):

    # --------------------------------------------------------
    # TRUE STATE
    # --------------------------------------------------------

    altitude = 600.0

    x = 0.0
    y = 0.0

    heading = 0.0

    dt = 0.1

    time = 0.0

    current_steering = 0.0

    next_guidance_update = 0.0

    # --------------------------------------------------------
    # INITIAL TRUE VELOCITY
    # --------------------------------------------------------

    vx_air = (
        horizontal_air_velocity *
        np.cos(heading)
    )

    vy_air = (
        horizontal_air_velocity *
        np.sin(heading)
    )

    vx = (
        vx_air +
        wind_x
    )

    vy = (
        vy_air +
        wind_y
    )

    true_state = np.array([
        x,
        y,
        altitude,
        heading,
        vx,
        vy
    ])

    # --------------------------------------------------------
    # EKF INITIALIZATION
    # --------------------------------------------------------

    estimated_state, covariance = (
        initialize_ekf(
            x,
            y,
            altitude,
            heading,
            vx,
            vy
        )
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    time_history = []

    true_x_history = []
    true_y_history = []

    estimated_x_history = []
    estimated_y_history = []

    true_altitude_history = []
    estimated_altitude_history = []

    true_heading_history = []
    estimated_heading_history = []

    position_error_history = []
    altitude_error_history = []
    heading_error_history = []

    steering_history = []

    # --------------------------------------------------------
    # GUIDANCE / SENSOR UPDATE RATES
    # --------------------------------------------------------

    next_gnss_update = 0.0
    next_baro_update = 0.0
    next_imu_update = 0.0

    GNSS_INTERVAL = 1.0
    BARO_INTERVAL = 0.2
    IMU_INTERVAL = 0.1

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    while altitude > 0:

        # ====================================================
        # TRUE PARAFOIL DYNAMICS
        # ====================================================

        true_turn_rate = (
            max_turn_rate *
            current_steering
        )

        heading += (
            true_turn_rate *
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

        vx = (
            vx_air +
            wind_x
        )

        vy = (
            vy_air +
            wind_y
        )

        x += (
            vx *
            dt
        )

        y += (
            vy *
            dt
        )

        altitude -= (
            vertical_velocity *
            dt
        )

        if altitude < 0:

            altitude = 0.0

        true_state = np.array([
            x,
            y,
            altitude,
            heading,
            vx,
            vy
        ])

        # ====================================================
        # EKF PREDICTION
        # ====================================================

        estimated_state, covariance = (
            ekf_predict(
                estimated_state,
                covariance,
                current_steering,
                wind_x,
                wind_y,
                dt
            )
        )

        # ====================================================
        # SENSOR MEASUREMENTS
        # ====================================================

        (
            measured_x,
            measured_y,
            measured_altitude,
            measured_heading,
            measured_vx,
            measured_vy,
            measured_turn_rate
        ) = generate_sensor_measurements(
            true_state,
            true_turn_rate
        )

        # ====================================================
        # GNSS UPDATE
        # ====================================================

        if time >= next_gnss_update:

            (
                estimated_state,
                covariance
            ) = ekf_gnss_update(
                estimated_state,
                covariance,
                measured_x,
                measured_y,
                measured_vx,
                measured_vy
            )

            next_gnss_update = (
                time +
                GNSS_INTERVAL
            )

        # ====================================================
        # BAROMETER UPDATE
        # ====================================================

        if time >= next_baro_update:

            (
                estimated_state,
                covariance
            ) = ekf_baro_update(
                estimated_state,
                covariance,
                measured_altitude
            )

            next_baro_update = (
                time +
                BARO_INTERVAL
            )

        # ====================================================
        # IMU UPDATE
        # ====================================================

        if time >= next_imu_update:

            (
                estimated_state,
                covariance
            ) = ekf_imu_update(
                estimated_state,
                covariance,
                measured_heading
            )

            next_imu_update = (
                time +
                IMU_INTERVAL
            )

        # ====================================================
        # GUIDANCE UPDATE
        # ====================================================

        if time >= next_guidance_update:

            (
                new_command,
                predicted_error,
                prediction_horizon
            ) = select_guidance_command(
                estimated_state,
                wind_x,
                wind_y
            )

            current_steering = (
                new_command
            )

            next_guidance_update = (
                time +
                guidance_interval
            )

        # ====================================================
        # ESTIMATION ERRORS
        # ====================================================

        position_error = np.sqrt(
            (
                true_state[0] -
                estimated_state[0]
            )**2
            +
            (
                true_state[1] -
                estimated_state[1]
            )**2
        )

        altitude_error = abs(
            true_state[2] -
            estimated_state[2]
        )

        heading_error = abs(
            normalize_angle(
                true_state[3] -
                estimated_state[3]
            )
        )

        # ====================================================
        # STORE HISTORY
        # ====================================================

        time_history.append(time)

        true_x_history.append(
            true_state[0]
        )

        true_y_history.append(
            true_state[1]
        )

        estimated_x_history.append(
            estimated_state[0]
        )

        estimated_y_history.append(
            estimated_state[1]
        )

        true_altitude_history.append(
            true_state[2]
        )

        estimated_altitude_history.append(
            estimated_state[2]
        )

        true_heading_history.append(
            np.degrees(
                true_state[3]
            )
        )

        estimated_heading_history.append(
            np.degrees(
                estimated_state[3]
            )
        )

        position_error_history.append(
            position_error
        )

        altitude_error_history.append(
            altitude_error
        )

        heading_error_history.append(
            np.degrees(
                heading_error
            )
        )

        steering_history.append(
            current_steering
        )

        time += dt

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    landing_error = np.sqrt(
        (
            x -
            target_x
        )**2
        +
        (
            y -
            target_y
        )**2
    )

    estimated_landing_error = np.sqrt(
        (
            estimated_state[0] -
            target_x
        )**2
        +
        (
            estimated_state[1] -
            target_y
        )**2
    )

    # ========================================================
    # ESTIMATION STATISTICS
    # ========================================================

    position_errors = np.array(
        position_error_history
    )

    altitude_errors = np.array(
        altitude_error_history
    )

    heading_errors = np.array(
        heading_error_history
    )

    steering_array = np.array(
        steering_history
    )

    mean_position_error = np.mean(
        position_errors
    )

    max_position_error = np.max(
        position_errors
    )

    rms_position_error = np.sqrt(
        np.mean(
            position_errors**2
        )
    )

    mean_altitude_error = np.mean(
        altitude_errors
    )

    rms_altitude_error = np.sqrt(
        np.mean(
            altitude_errors**2
        )
    )

    mean_heading_error = np.mean(
        heading_errors
    )

    rms_heading_error = np.sqrt(
        np.mean(
            heading_errors**2
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

        "landing_error":
            landing_error,

        "estimated_landing_error":
            estimated_landing_error,

        "landing_x":
            x,

        "landing_y":
            y,

        "estimated_x":
            estimated_state[0],

        "estimated_y":
            estimated_state[1],

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

        "mean_heading_error":
            mean_heading_error,

        "rms_heading_error":
            rms_heading_error,

        "average_steering":
            average_steering,

        "steering_reversals":
            steering_reversals,

        "time_history":
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

        "position_error_history":
            position_error_history,

        "altitude_error_history":
            altitude_error_history,

        "heading_error_history":
            heading_error_history,

        "steering_history":
            steering_history
    }


# ============================================================
# PRINT SYSTEM PARAMETERS
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
# PRINT SENSOR PARAMETERS
# ============================================================

print()
print("========================================")
print("SENSOR MODEL")
print("========================================")

print(
    "GNSS position noise:",
    GNSS_POSITION_STD,
    "m"
)

print(
    "GNSS velocity noise:",
    GNSS_VELOCITY_STD,
    "m/s"
)

print(
    "Barometer altitude noise:",
    BARO_ALTITUDE_STD,
    "m"
)

print(
    "IMU heading noise:",
    np.degrees(
        IMU_HEADING_STD
    ),
    "deg"
)

print(
    "IMU turn-rate noise:",
    np.degrees(
        IMU_TURN_RATE_STD
    ),
    "deg/s"
)


# ============================================================
# TARGET
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
# REFERENCE WIND
# ============================================================

wind_speed = 3.0
wind_direction_deg = 0.0

wind_direction = np.radians(
    wind_direction_deg
)

wind_x = (
    wind_speed *
    np.cos(
        wind_direction
    )
)

wind_y = (
    wind_speed *
    np.sin(
        wind_direction
    )
)


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
# GUIDANCE INFORMATION
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
    len(
        candidate_commands
    )
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
# RUN SIMULATION
# ============================================================

print()
print("========================================")
print("STARTING V10.0 SIMULATION")
print("========================================")

np.random.seed(42)

simulation = run_simulation(
    wind_x,
    wind_y
)


# ============================================================
# EXTRACT RESULTS
# ============================================================

landing_error = simulation[
    "landing_error"
]

estimated_landing_error = simulation[
    "estimated_landing_error"
]

landing_x = simulation[
    "landing_x"
]

landing_y = simulation[
    "landing_y"
]

estimated_x = simulation[
    "estimated_x"
]

estimated_y = simulation[
    "estimated_y"
]

flight_time = simulation[
    "flight_time"
]

mean_position_error = simulation[
    "mean_position_error"
]

rms_position_error = simulation[
    "rms_position_error"
]

max_position_error = simulation[
    "max_position_error"
]

mean_altitude_error = simulation[
    "mean_altitude_error"
]

rms_altitude_error = simulation[
    "rms_altitude_error"
]

mean_heading_error = simulation[
    "mean_heading_error"
]

rms_heading_error = simulation[
    "rms_heading_error"
]

average_steering = simulation[
    "average_steering"
]

steering_reversals = simulation[
    "steering_reversals"
]


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("========================================")
print("V10.0 FINAL RESULTS")
print("========================================")

print(
    "True landing X:",
    landing_x,
    "m"
)

print(
    "True landing Y:",
    landing_y,
    "m"
)

print(
    "Estimated landing X:",
    estimated_x,
    "m"
)

print(
    "Estimated landing Y:",
    estimated_y,
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
    landing_error,
    "m"
)

print(
    "Estimated landing error:",
    estimated_landing_error,
    "m"
)

print(
    "Flight time:",
    flight_time,
    "s"
)

print(
    "Average steering:",
    average_steering
)

print(
    "Steering reversals:",
    steering_reversals
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
    mean_position_error,
    "m"
)

print(
    "RMS position estimation error:",
    rms_position_error,
    "m"
)

print(
    "Maximum position estimation error:",
    max_position_error,
    "m"
)

print(
    "Mean altitude estimation error:",
    mean_altitude_error,
    "m"
)

print(
    "RMS altitude estimation error:",
    rms_altitude_error,
    "m"
)

print(
    "Mean heading estimation error:",
    mean_heading_error,
    "degrees"
)

print(
    "RMS heading estimation error:",
    rms_heading_error,
    "degrees"
)


# ============================================================
# SENSOR VS EKF COMPARISON
# ============================================================

print()
print("========================================")
print("NAVIGATION SYSTEM ASSESSMENT")
print("========================================")

print(
    "Raw GNSS position noise:",
    GNSS_POSITION_STD,
    "m"
)

print(
    "EKF RMS position error:",
    rms_position_error,
    "m"
)

if rms_position_error < GNSS_POSITION_STD:

    print(
        "EKF STATUS: ESTIMATION IMPROVED"
    )

else:

    print(
        "EKF STATUS: REQUIRES TUNING"
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
    label="EKF estimated trajectory"
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
    "V10.0 True vs EKF Estimated Trajectory"
)

plt.legend()

plt.grid()

plt.axis("equal")

plt.show()


# ============================================================
# PLOT 2: POSITION ESTIMATION ERROR
# ============================================================

plt.figure()

plt.plot(
    simulation[
        "time_history"
    ],
    simulation[
        "position_error_history"
    ]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Position Estimation Error (m)"
)

plt.title(
    "V10.0 EKF Position Estimation Error"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 3: TRUE VS ESTIMATED ALTITUDE
# ============================================================

plt.figure()

plt.plot(
    simulation[
        "time_history"
    ],
    simulation[
        "true_altitude_history"
    ],
    label="True altitude"
)

plt.plot(
    simulation[
        "time_history"
    ],
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
    "V10.0 True vs EKF Estimated Altitude"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# PLOT 4: TRUE VS ESTIMATED HEADING
# ============================================================

plt.figure()

plt.plot(
    simulation[
        "time_history"
    ],
    simulation[
        "true_heading_history"
    ],
    label="True heading"
)

plt.plot(
    simulation[
        "time_history"
    ],
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
    "Heading (degrees)"
)

plt.title(
    "V10.0 True vs EKF Estimated Heading"
)

plt.legend()

plt.grid()

plt.show()


# ============================================================
# PLOT 5: STEERING COMMAND
# ============================================================

plt.figure()

plt.step(
    simulation[
        "time_history"
    ],
    simulation[
        "steering_history"
    ],
    where="post"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Steering Command"
)

plt.title(
    "V10.0 EKF-Based Guidance Steering Command"
)

plt.grid()

plt.show()


# ============================================================
# COMPLETION SUMMARY
# ============================================================

print()
print("========================================")
print("V10.0 SIMULATION COMPLETE")
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
    "GNSS position noise:",
    GNSS_POSITION_STD,
    "m"
)

print(
    "EKF RMS position error:",
    rms_position_error,
    "m"
)

print(
    "True landing error:",
    landing_error,
    "m"
)

print(
    "Estimated landing error:",
    estimated_landing_error,
    "m"
)

print()
print(
    "NEXT DEVELOPMENT STEP:"
)

print(
    "V10.1 -> EKF tuning + sensor update-rate study"
)

print("========================================")