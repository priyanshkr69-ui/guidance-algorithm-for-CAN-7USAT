import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V8.6
# 2D WIND DIRECTION SENSITIVITY
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
# WIND TEST PARAMETERS
# ============================================================

wind_speed = 3.0

wind_directions_deg = [
    0.0,
    45.0,
    90.0,
    135.0,
    180.0,
    225.0,
    270.0,
    315.0
]


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
# PRINT PARAMETERS
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V8.6")
print("2D WIND DIRECTION SENSITIVITY")
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

print("WIND TEST")

print(
    "Wind speed:",
    wind_speed,
    "m/s"
)

print(
    "Wind directions:",
    wind_directions_deg
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

print("========================================")


# ============================================================
# FUNCTION:
# CONVERT WIND SPEED + DIRECTION TO X/Y
# ============================================================

def calculate_wind_components(
    speed,
    direction_deg
):

    direction_rad = np.radians(
        direction_deg
    )

    wind_x = (
        speed *
        np.cos(direction_rad)
    )

    wind_y = (
        speed *
        np.sin(direction_rad)
    )

    return wind_x, wind_y


# ============================================================
# FUNCTION:
# RUN ONE SIMULATION
# ============================================================

def run_simulation(
    wind_x,
    wind_y
):

    # --------------------------------------------------------
    # INITIAL CONDITIONS
    # --------------------------------------------------------

    altitude = 600.0

    x = 0.0
    y = 0.0

    heading = np.radians(
        0.0
    )

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

        # ----------------------------------------------------
        # Remaining flight time
        # ----------------------------------------------------

        remaining_time = (
            current_altitude /
            vertical_velocity
        )


        # ----------------------------------------------------
        # Limit prediction to remaining flight
        # ----------------------------------------------------

        actual_prediction_time = min(
            prediction_horizon,
            remaining_time
        )


        prediction_dt = 0.5

        steps = int(
            actual_prediction_time /
            prediction_dt
        )


        # ----------------------------------------------------
        # Copy state
        # ----------------------------------------------------

        px = current_x
        py = current_y

        pheading = current_heading


        # ----------------------------------------------------
        # Future trajectory
        # ----------------------------------------------------

        for _ in range(steps):

            # -----------------------------------------------
            # Turn rate
            # -----------------------------------------------

            turn_rate = (
                max_turn_rate *
                steering_command
            )


            # -----------------------------------------------
            # Heading update
            # -----------------------------------------------

            pheading += (
                turn_rate *
                prediction_dt
            )


            # -----------------------------------------------
            # Normalize heading
            # -----------------------------------------------

            pheading = (
                pheading + np.pi
            ) % (
                2 * np.pi
            ) - np.pi


            # -----------------------------------------------
            # Air velocity
            # -----------------------------------------------

            vx_air = (
                horizontal_air_velocity *
                np.cos(pheading)
            )

            vy_air = (
                horizontal_air_velocity *
                np.sin(pheading)
            )


            # -----------------------------------------------
            # Ground velocity
            # -----------------------------------------------

            vx_ground = (
                vx_air +
                wind_x
            )

            vy_ground = (
                vy_air +
                wind_y
            )


            # -----------------------------------------------
            # Position update
            # -----------------------------------------------

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
            # Adaptive prediction horizon
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

            best_error = float(
                "inf"
            )

            best_predicted_x = x

            best_predicted_y = y


            # ------------------------------------------------
            # Test candidate steering commands
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
                # Predicted distance to target
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

                    best_predicted_x = (
                        predicted_x
                    )

                    best_predicted_y = (
                        predicted_y
                    )


            # ------------------------------------------------
            # Apply selected command
            # ------------------------------------------------

            current_steering = (
                best_command
            )

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
            # Schedule next guidance update
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
        # Heading update
        # ----------------------------------------------------

        heading += (
            turn_rate *
            dt
        )


        # ----------------------------------------------------
        # Normalize heading
        # ----------------------------------------------------

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
            np.degrees(
                heading
            )
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


    # --------------------------------------------------------
    # Maximum steering
    # --------------------------------------------------------

    max_steering = np.max(
        np.abs(
            steering_array
        )
    )


    # --------------------------------------------------------
    # Average absolute steering
    # --------------------------------------------------------

    average_steering = np.mean(
        np.abs(
            steering_array
        )
    )


    # --------------------------------------------------------
    # Steering reversals
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
# RUN WIND-DIRECTION TESTS
# ============================================================

results = []


for direction_deg in wind_directions_deg:


    # --------------------------------------------------------
    # Calculate wind components
    # --------------------------------------------------------

    wind_x, wind_y = (
        calculate_wind_components(
            wind_speed,
            direction_deg
        )
    )


    print()
    print("----------------------------------------")

    print(
        "Running simulation for wind direction:",
        direction_deg,
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


    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    result = run_simulation(
        wind_x,
        wind_y
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


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

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
    "V8.6 WIND DIRECTION SENSITIVITY RESULTS"
)

print(
    "============================================================"
)


print(
    f"{'Direction':<15}"
    f"{'Wind X':<15}"
    f"{'Wind Y':<15}"
    f"{'Landing Error':<20}"
    f"{'Avg Steering':<16}"
    f"{'Reversals':<12}"
)

print(
    "------------------------------------------------------------"
)


for direction, result in zip(
    wind_directions_deg,
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


    wind_x, wind_y = (
        calculate_wind_components(
            wind_speed,
            direction
        )
    )


    print(
        f"{direction:<15.1f}"
        f"{wind_x:<15.3f}"
        f"{wind_y:<15.3f}"
        f"{landing_error:<20.3f}"
        f"{average_steering:<16.3f}"
        f"{steering_reversals:<12}"
    )


print(
    "============================================================"
)


# ============================================================
# FIND BEST AND WORST DIRECTIONS
# ============================================================

landing_errors = [
    result[0]
    for result in results
]


best_index = np.argmin(
    landing_errors
)

worst_index = np.argmax(
    landing_errors
)


print()
print(
    "========================================"
)

print(
    "BEST WIND DIRECTION"
)

print(
    "Direction:",
    wind_directions_deg[
        best_index
    ],
    "degrees"
)

print(
    "Landing error:",
    landing_errors[
        best_index
    ],
    "m"
)

print(
    "========================================"
)


print()
print(
    "========================================"
)

print(
    "WORST WIND DIRECTION"
)

print(
    "Direction:",
    wind_directions_deg[
        worst_index
    ],
    "degrees"
)

print(
    "Landing error:",
    landing_errors[
        worst_index
    ],
    "m"
)

print(
    "========================================"
)


# ============================================================
# PLOT 1:
# LANDING ERROR VS WIND DIRECTION
# ============================================================

plt.figure()


plt.plot(
    wind_directions_deg,
    landing_errors,
    marker="o"
)


plt.xlabel(
    "Wind Direction (degrees)"
)

plt.ylabel(
    "Landing Error (m)"
)

plt.title(
    "V8.6 Landing Error vs Wind Direction"
)

plt.grid()

plt.xticks(
    wind_directions_deg
)

plt.show()


# ============================================================
# PLOT 2:
# AVERAGE STEERING VS WIND DIRECTION
# ============================================================

average_steerings = [
    result[5]
    for result in results
]


plt.figure()


plt.plot(
    wind_directions_deg,
    average_steerings,
    marker="o"
)


plt.xlabel(
    "Wind Direction (degrees)"
)

plt.ylabel(
    "Average Absolute Steering"
)

plt.title(
    "V8.6 Average Steering vs Wind Direction"
)

plt.grid()

plt.xticks(
    wind_directions_deg
)

plt.show()


# ============================================================
# PLOT 3:
# LANDING POSITIONS
# ============================================================

landing_x_values = [
    result[2]
    for result in results
]

landing_y_values = [
    result[3]
    for result in results
]


plt.figure()


plt.scatter(
    landing_x_values,
    landing_y_values,
    label="Landing positions"
)


plt.scatter(
    target_x,
    target_y,
    label="Target"
)


for i, direction in enumerate(
    wind_directions_deg
):

    plt.annotate(
        f"{direction:.0f}°",
        (
            landing_x_values[i],
            landing_y_values[i]
        )
    )


plt.xlabel(
    "Landing X (m)"
)

plt.ylabel(
    "Landing Y (m)"
)

plt.title(
    "V8.6 Landing Position Under Different Wind Directions"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 4:
# TRAJECTORIES FOR ALL WIND DIRECTIONS
# ============================================================

plt.figure()


for i, direction in enumerate(
    wind_directions_deg
):


    result = results[i]


    x_history = result[8]
    y_history = result[9]


    plt.plot(
        x_history,
        y_history,
        label=f"{direction:.0f}°"
    )


plt.scatter(
    target_x,
    target_y,
    label="Target"
)


plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "V8.6 Parafoil Trajectories - 3 m/s Wind"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 5:
# STEERING REVERSALS VS WIND DIRECTION
# ============================================================

steering_reversals_values = [
    result[6]
    for result in results
]


plt.figure()


plt.plot(
    wind_directions_deg,
    steering_reversals_values,
    marker="o"
)


plt.xlabel(
    "Wind Direction (degrees)"
)

plt.ylabel(
    "Steering Reversals"
)

plt.title(
    "V8.6 Steering Reversals vs Wind Direction"
)

plt.grid()

plt.xticks(
    wind_directions_deg
)

plt.show()


# ============================================================
# REFERENCE CASE:
# 0 DEGREE WIND
# ============================================================

reference_index = 0

reference_result = results[
    reference_index
]


(
    reference_landing_error,
    reference_flight_time,
    reference_landing_x,
    reference_landing_y,
    reference_max_steering,
    reference_average_steering,
    reference_reversals,
    reference_time_history,
    reference_x_history,
    reference_y_history,
    reference_altitude_history,
    reference_heading_history,
    reference_steering_history,
    reference_prediction_error_history,
    reference_prediction_horizon_history,
    reference_actual_distance_history
) = reference_result


# ============================================================
# REFERENCE TRAJECTORY
# ============================================================

plt.figure()


plt.plot(
    reference_x_history,
    reference_y_history,
    label="Actual trajectory"
)


plt.scatter(
    reference_x_history[0],
    reference_y_history[0],
    label="Deployment"
)


plt.scatter(
    target_x,
    target_y,
    label="Target"
)


plt.scatter(
    reference_x_history[-1],
    reference_y_history[-1],
    label="Landing"
)


plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "V8.6 Reference Trajectory - 0° Wind"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print(
    "========================================"
)

print(
    "V8.6 SIMULATION COMPLETE"
)

print(
    "========================================"
)

print(
    "Wind speed:",
    wind_speed,
    "m/s"
)

print(
    "Number of wind directions tested:",
    len(wind_directions_deg)
)

print(
    "Best direction:",
    wind_directions_deg[
        best_index
    ],
    "degrees"
)

print(
    "Best landing error:",
    landing_errors[
        best_index
    ],
    "m"
)

print(
    "Worst direction:",
    wind_directions_deg[
        worst_index
    ],
    "degrees"
)

print(
    "Worst landing error:",
    landing_errors[
        worst_index
    ],
    "m"
)

print(
    "========================================"
)