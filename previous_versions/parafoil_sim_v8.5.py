import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V8.5
# ADAPTIVE HORIZON GUIDANCE
# ============================================================


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


# ============================================================
# CONTROL PARAMETERS
# ============================================================

max_turn_rate = np.radians(15.0)

# How frequently guidance selects a new command
guidance_interval = 2.0

# Candidate steering commands
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
# PRINT PARAFOIL PARAMETERS
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V8.5")
print("ADAPTIVE HORIZON GUIDANCE")
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

print("----------------------------------------")

print("TARGET")

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

print("----------------------------------------")

print("GUIDANCE")

print(
    "Guidance interval:",
    guidance_interval,
    "s"
)

print(
    "Candidate commands:",
    len(candidate_commands)
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

print("========================================")


# ============================================================
# FUNCTION: RUN ONE SIMULATION
# ============================================================

def run_simulation(wind_x, wind_y=0.0):


    # --------------------------------------------------------
    # INITIAL CONDITIONS
    # --------------------------------------------------------

    altitude = 600.0

    x = 0.0
    y = 0.0

    heading = np.radians(0.0)

    dt = 0.1

    time = 0.0

    next_guidance_update = 0.0

    current_steering = 0.0


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    time_history = []

    x_history = []
    y_history = []

    altitude_history = []

    heading_history = []

    steering_history = []

    prediction_error_history = []

    prediction_horizon_history = []

    actual_distance_history = []


    # --------------------------------------------------------
    # CURRENT PREDICTION VALUES
    # --------------------------------------------------------

    predicted_x_current = x
    predicted_y_current = y

    predicted_error_current = 0.0

    current_prediction_horizon = 20.0


    # ========================================================
    # PREDICT FUTURE TRAJECTORY
    # ========================================================

    def predict_landing(
        current_x,
        current_y,
        current_altitude,
        current_heading,
        steering_command,
        prediction_horizon
    ):

        # Remaining flight time

        remaining_time = (
            current_altitude /
            vertical_velocity
        )


        # Never predict beyond actual
        # remaining flight time

        actual_prediction_time = min(
            prediction_horizon,
            remaining_time
        )


        prediction_dt = 0.5

        steps = int(
            actual_prediction_time /
            prediction_dt
        )


        # Copy current state

        px = current_x
        py = current_y

        pheading = current_heading


        # ----------------------------------------------------
        # Future trajectory prediction
        # ----------------------------------------------------

        for _ in range(steps):


            # Turn rate

            turn_rate = (
                max_turn_rate *
                steering_command
            )


            # Heading update

            pheading += (
                turn_rate *
                prediction_dt
            )


            # Normalize heading

            pheading = (
                pheading + np.pi
            ) % (
                2 * np.pi
            ) - np.pi


            # ------------------------------------------------
            # Air velocity
            # ------------------------------------------------

            vx_air = (
                horizontal_air_velocity *
                np.cos(pheading)
            )

            vy_air = (
                horizontal_air_velocity *
                np.sin(pheading)
            )


            # ------------------------------------------------
            # Ground velocity
            # ------------------------------------------------

            vx_ground = (
                vx_air +
                wind_x
            )

            vy_ground = (
                vy_air +
                wind_y
            )


            # ------------------------------------------------
            # Position update
            # ------------------------------------------------

            px += (
                vx_ground *
                prediction_dt
            )

            py += (
                vy_ground *
                prediction_dt
            )


        return px, py


    # ========================================================
    # MAIN SIMULATION LOOP
    # ========================================================

    while altitude > 0:


        # ====================================================
        # GUIDANCE UPDATE
        # ====================================================

        if time >= next_guidance_update:


            # ------------------------------------------------
            # Select adaptive prediction horizon
            # ------------------------------------------------

            current_prediction_horizon = (
                get_prediction_horizon(
                    altitude
                )
            )


            # ------------------------------------------------
            # Initial best solution
            # ------------------------------------------------

            best_command = 0.0

            best_error = float("inf")

            best_predicted_x = x

            best_predicted_y = y


            # ------------------------------------------------
            # Test every steering command
            # ------------------------------------------------

            for command in candidate_commands:


                predicted_x, predicted_y = (
                    predict_landing(
                        x,
                        y,
                        altitude,
                        heading,
                        command,
                        current_prediction_horizon
                    )
                )


                # --------------------------------------------
                # Distance between predicted position
                # and target
                # --------------------------------------------

                error = np.sqrt(
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


                # --------------------------------------------
                # Select best command
                # --------------------------------------------

                if error < best_error:

                    best_error = error

                    best_command = command

                    best_predicted_x = predicted_x

                    best_predicted_y = predicted_y


            # ------------------------------------------------
            # Apply best steering command
            # ------------------------------------------------

            current_steering = best_command

            predicted_x_current = (
                best_predicted_x
            )

            predicted_y_current = (
                best_predicted_y
            )

            predicted_error_current = (
                best_error
            )


            # ------------------------------------------------
            # Next guidance update
            # ------------------------------------------------

            next_guidance_update = (
                time +
                guidance_interval
            )


        # ====================================================
        # ACTUAL PARAFOIL DYNAMICS
        # ====================================================

        turn_rate = (
            max_turn_rate *
            current_steering
        )


        # ----------------------------------------------------
        # Update heading
        # ----------------------------------------------------

        heading += (
            turn_rate *
            dt
        )


        heading = (
            heading + np.pi
        ) % (
            2 * np.pi
        ) - np.pi


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

        vx_ground = (
            vx_air +
            wind_x
        )

        vy_ground = (
            vy_air +
            wind_y
        )


        # ----------------------------------------------------
        # Position update
        # ----------------------------------------------------

        x += (
            vx_ground *
            dt
        )

        y += (
            vy_ground *
            dt
        )


        # ----------------------------------------------------
        # Altitude update
        # ----------------------------------------------------

        altitude -= (
            vertical_velocity *
            dt
        )


        # ====================================================
        # STORE HISTORY
        # ====================================================

        time_history.append(
            time
        )

        x_history.append(
            x
        )

        y_history.append(
            y
        )

        altitude_history.append(
            altitude
        )

        heading_history.append(
            np.degrees(heading)
        )

        steering_history.append(
            current_steering
        )

        prediction_error_history.append(
            predicted_error_current
        )

        prediction_horizon_history.append(
            current_prediction_horizon
        )

        actual_distance_history.append(
            np.sqrt(
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
        )


        # ====================================================
        # TIME UPDATE
        # ====================================================

        time += dt


    # ========================================================
    # FINAL LANDING ERROR
    # ========================================================

    landing_error = np.sqrt(
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


    # ========================================================
    # CONTROLLER ANALYSIS
    # ========================================================

    steering_array = np.array(
        steering_history
    )


    # Maximum steering

    max_steering = np.max(
        np.abs(
            steering_array
        )
    )


    # Average steering

    average_steering = np.mean(
        np.abs(
            steering_array
        )
    )


    # --------------------------------------------------------
    # Count steering reversals
    # --------------------------------------------------------

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


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return (
        landing_error,
        time,
        x,
        y,
        max_steering,
        average_steering,
        steering_reversals,
        time_history,
        x_history,
        y_history,
        altitude_history,
        heading_history,
        steering_history,
        prediction_error_history,
        prediction_horizon_history,
        actual_distance_history
    )


# ============================================================
# WIND TESTS
# ============================================================

wind_values = [
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


# ============================================================
# RUN ALL WIND CONDITIONS
# ============================================================

for wind_x in wind_values:


    print()
    print("----------------------------------------")

    print(
        "Running simulation for wind:",
        wind_x,
        "m/s"
    )


    result = run_simulation(
        wind_x
    )


    (
        landing_error,
        flight_time,
        landing_x,
        landing_y,
        max_steering,
        average_steering,
        steering_reversals,
        time_history,
        x_history,
        y_history,
        altitude_history,
        heading_history,
        steering_history,
        prediction_error_history,
        prediction_horizon_history,
        actual_distance_history
    ) = result


    results.append(
        result
    )


    print(
        "Landing X:",
        landing_x,
        "m"
    )

    print(
        "Landing Y:",
        landing_y,
        "m"
    )

    print(
        "Landing error:",
        landing_error,
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
# FINAL RESULTS TABLE
# ============================================================

print()
print()
print(
    "============================================================"
)

print(
    "V8.5 ADAPTIVE HORIZON RESULTS"
)

print(
    "============================================================"
)


print(
    f"{'Wind (m/s)':<15}"
    f"{'Landing Error (m)':<20}"
    f"{'Flight Time (s)':<18}"
    f"{'Avg Steering':<16}"
    f"{'Reversals':<12}"
)

print(
    "------------------------------------------------------------"
)


for wind, result in zip(
    wind_values,
    results
):


    (
        landing_error,
        flight_time,
        landing_x,
        landing_y,
        max_steering,
        average_steering,
        steering_reversals,
        *_ 
    ) = result


    print(
        f"{wind:<15.1f}"
        f"{landing_error:<20.3f}"
        f"{flight_time:<18.2f}"
        f"{average_steering:<16.3f}"
        f"{steering_reversals:<12}"
    )


print(
    "============================================================"
)


# ============================================================
# SELECT 3 m/s CASE FOR DETAILED PLOTS
# ============================================================

reference_wind = 3.0

reference_index = wind_values.index(
    reference_wind
)

reference_result = results[
    reference_index
]


(
    landing_error,
    flight_time,
    landing_x,
    landing_y,
    max_steering,
    average_steering,
    steering_reversals,
    time_history,
    x_history,
    y_history,
    altitude_history,
    heading_history,
    steering_history,
    prediction_error_history,
    prediction_horizon_history,
    actual_distance_history
) = reference_result


# ============================================================
# REFERENCE CASE INFORMATION
# ============================================================

print()
print(
    "========================================"
)

print(
    "REFERENCE CASE:"
)

print(
    "Wind =",
    reference_wind,
    "m/s"
)

print(
    "Landing error =",
    landing_error,
    "m"
)

print(
    "========================================"
)


# ============================================================
# PLOT 1:
# LANDING ERROR VS WIND SPEED
# ============================================================

landing_errors = [
    result[0]
    for result in results
]


plt.figure()


plt.plot(
    wind_values,
    landing_errors,
    marker="o"
)


plt.xlabel(
    "Wind Speed (m/s)"
)

plt.ylabel(
    "Landing Error (m)"
)

plt.title(
    "V8.5 Landing Error vs Wind Speed"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 2:
# AVERAGE STEERING VS WIND SPEED
# ============================================================

average_steerings = [
    result[5]
    for result in results
]


plt.figure()


plt.plot(
    wind_values,
    average_steerings,
    marker="o"
)


plt.xlabel(
    "Wind Speed (m/s)"
)

plt.ylabel(
    "Average Absolute Steering"
)

plt.title(
    "V8.5 Average Steering vs Wind Speed"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 3:
# ACTUAL TRAJECTORY
# ============================================================

plt.figure()


plt.plot(
    x_history,
    y_history,
    label="Actual trajectory"
)


plt.scatter(
    x_history[0],
    y_history[0],
    label="Deployment"
)


plt.scatter(
    target_x,
    target_y,
    label="Target"
)


plt.scatter(
    x_history[-1],
    y_history[-1],
    label="Landing"
)


plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "V8.5 Parafoil Trajectory - 3 m/s Wind"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 4:
# ACTUAL DISTANCE TO TARGET
# ============================================================

plt.figure()


plt.plot(
    time_history,
    actual_distance_history
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Distance to Target (m)"
)

plt.title(
    "V8.5 Distance to Target - 3 m/s Wind"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 5:
# STEERING COMMAND
# ============================================================

plt.figure()


plt.step(
    time_history,
    steering_history,
    where="post"
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Steering Command"
)

plt.title(
    "V8.5 Steering Command - 3 m/s Wind"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 6:
# HEADING
# ============================================================

plt.figure()


plt.plot(
    time_history,
    heading_history
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Heading (degrees)"
)

plt.title(
    "V8.5 Parafoil Heading - 3 m/s Wind"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 7:
# ALTITUDE
# ============================================================

plt.figure()


plt.plot(
    time_history,
    altitude_history
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Altitude (m)"
)

plt.title(
    "V8.5 Altitude During Descent"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 8:
# PREDICTION ERROR
# ============================================================

plt.figure()


plt.plot(
    time_history,
    prediction_error_history
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Predicted Position Error (m)"
)

plt.title(
    "V8.5 Predicted Error - 3 m/s Wind"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 9:
# ADAPTIVE PREDICTION HORIZON
# ============================================================

plt.figure()


plt.step(
    time_history,
    prediction_horizon_history,
    where="post"
)


plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Prediction Horizon (s)"
)

plt.title(
    "V8.5 Adaptive Prediction Horizon"
)

plt.grid()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print(
    "========================================"
)

print(
    "V8.5 SIMULATION COMPLETE"
)

print(
    "========================================"
)

print(
    "Reference wind:",
    reference_wind,
    "m/s"
)

print(
    "Final landing position:",
    landing_x,
    ",",
    landing_y,
    "m"
)

print(
    "Final landing error:",
    landing_error,
    "m"
)

print(
    "========================================"
)