import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V8.3
# STEERING COMMAND DIAGNOSTIC
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

guidance_interval = 2.0

candidate_commands = np.linspace(
    -1.0,
    1.0,
    21
)


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

    steering_history = []


    # --------------------------------------------------------
    # PREDICT LANDING FUNCTION
    # --------------------------------------------------------

    def predict_landing(
        current_x,
        current_y,
        current_altitude,
        current_heading,
        steering_command
    ):

        remaining_time = (
            current_altitude /
            vertical_velocity
        )

        prediction_dt = 0.5

        steps = int(
            remaining_time /
            prediction_dt
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

            pheading = (
                pheading + np.pi
            ) % (2 * np.pi) - np.pi

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

        return px, py


    # ========================================================
    # MAIN SIMULATION LOOP
    # ========================================================

    while altitude > 0:

        # ----------------------------------------------------
        # GUIDANCE UPDATE
        # ----------------------------------------------------

        if time >= next_guidance_update:

            best_command = 0.0

            best_error = float("inf")

            best_predicted_x = x

            best_predicted_y = y


            # =================================================
            # TEST ALL CANDIDATE COMMANDS
            # =================================================

            candidate_errors = []


            for command in candidate_commands:

                predicted_x, predicted_y = predict_landing(
                    x,
                    y,
                    altitude,
                    heading,
                    command
                )


                error = np.sqrt(
                    (predicted_x - target_x)**2 +
                    (predicted_y - target_y)**2
                )


                candidate_errors.append(
                    (
                        command,
                        predicted_x,
                        predicted_y,
                        error
                    )
                )


                # Keep best command

                if error < best_error:

                    best_error = error

                    best_command = command

                    best_predicted_x = predicted_x

                    best_predicted_y = predicted_y


            # =================================================
            # 4 m/s WIND DIAGNOSTIC
            # =================================================

            if (
                abs(wind_x - 4.0) < 0.001
                and time < 0.01
            ):

                print()
                print()
                print("========================================")
                print("4 m/s WIND - CANDIDATE ANALYSIS")
                print("========================================")

                print(
                    "Current time:",
                    time,
                    "s"
                )

                print(
                    "Current position:",
                    x,
                    ",",
                    y,
                    "m"
                )

                print(
                    "Current altitude:",
                    altitude,
                    "m"
                )

                print(
                    "Current heading:",
                    np.degrees(heading),
                    "degrees"
                )

                print("----------------------------------------")

                print(
                    f"{'Steering':<12}"
                    f"{'Pred. X':<15}"
                    f"{'Pred. Y':<15}"
                    f"{'Error (m)':<15}"
                )

                print("----------------------------------------")


                for (
                    command,
                    predicted_x,
                    predicted_y,
                    error
                ) in candidate_errors:

                    print(
                        f"{command:<12.1f}"
                        f"{predicted_x:<15.2f}"
                        f"{predicted_y:<15.2f}"
                        f"{error:<15.2f}"
                    )


                print("----------------------------------------")

                print(
                    "SELECTED COMMAND:",
                    best_command
                )

                print(
                    "BEST PREDICTED LANDING:",
                    best_predicted_x,
                    ",",
                    best_predicted_y
                )

                print(
                    "BEST PREDICTED ERROR:",
                    best_error,
                    "m"
                )

                print("========================================")
                print()


            # ------------------------------------------------
            # APPLY BEST COMMAND
            # ------------------------------------------------

            current_steering = best_command


            # Next guidance update

            next_guidance_update = (
                time +
                guidance_interval
            )


        # ====================================================
        # PARAFOIL DYNAMICS
        # ====================================================

        turn_rate = (
            max_turn_rate *
            current_steering
        )


        heading += (
            turn_rate *
            dt
        )


        heading = (
            heading + np.pi
        ) % (2 * np.pi) - np.pi


        # ----------------------------------------------------
        # AIR VELOCITY
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
        # GROUND VELOCITY
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
        # POSITION
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
        # ALTITUDE
        # ----------------------------------------------------

        altitude -= (
            vertical_velocity *
            dt
        )


        # ----------------------------------------------------
        # STORE HISTORY
        # ----------------------------------------------------

        time_history.append(time)

        x_history.append(x)

        y_history.append(y)

        steering_history.append(
            current_steering
        )


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        time += dt


    # ========================================================
    # FINAL LANDING ERROR
    # ========================================================

    landing_error = np.sqrt(
        (x - target_x)**2 +
        (y - target_y)**2
    )


    # ========================================================
    # CONTROLLER ANALYSIS
    # ========================================================

    steering_array = np.array(
        steering_history
    )


    max_steering = np.max(
        np.abs(steering_array)
    )


    average_steering = np.mean(
        np.abs(steering_array)
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


    return (
        landing_error,
        time,
        x,
        y,
        max_steering,
        average_steering,
        steering_reversals
    )


# ============================================================
# PRINT PARAFOIL PARAMETERS
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V8.3")
print("STEERING COMMAND DIAGNOSTIC")
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

print("========================================")


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
        steering_reversals
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
# FINAL WIND SENSITIVITY TABLE
# ============================================================

print()
print()

print("============================================================")
print("V8.3 WIND SENSITIVITY RESULTS")
print("============================================================")

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
        steering_reversals
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
# PLOT: LANDING ERROR VS WIND
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
    "V8.3 Landing Error vs Wind Speed"
)

plt.grid()

plt.show()


# ============================================================
# PLOT: AVERAGE STEERING VS WIND
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
    "V8.3 Average Steering vs Wind Speed"
)

plt.grid()

plt.show()