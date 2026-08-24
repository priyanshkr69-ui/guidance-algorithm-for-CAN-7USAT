import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V8.7
# MONTE CARLO WIND ROBUSTNESS ANALYSIS
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
# RUN ONE SIMULATION
# ============================================================

def run_simulation(
    wind_speed,
    wind_direction_deg
):

    # --------------------------------------------------------
    # Convert wind direction to components
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Initial conditions
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
    # History
    # --------------------------------------------------------

    steering_history = []

    x_history = []
    y_history = []

    time_history = []


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

        # Prediction time cannot exceed
        # remaining flight time

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


        # Copy current state

        px = current_x
        py = current_y
        pheading = current_heading


        # ----------------------------------------------------
        # Future trajectory
        # ----------------------------------------------------

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
            ) % (
                2 * np.pi
            ) - np.pi


            # Air velocity

            vx_air = (
                horizontal_air_velocity *
                np.cos(pheading)
            )

            vy_air = (
                horizontal_air_velocity *
                np.sin(pheading)
            )


            # Ground velocity

            vx_ground = (
                vx_air +
                wind_x
            )

            vy_ground = (
                vy_air +
                wind_y
            )


            # Position update

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


            best_command = 0.0

            best_error = float("inf")


            # Adaptive horizon

            prediction_horizon = (
                get_prediction_horizon(
                    altitude
                )
            )


            # ------------------------------------------------
            # Test every candidate steering command
            # ------------------------------------------------

            for command in candidate_commands:


                predicted_x, predicted_y = (
                    predict_landing(
                        x,
                        y,
                        altitude,
                        heading,
                        command,
                        prediction_horizon
                    )
                )


                # Landing prediction error

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


                # Select best command

                if error < best_error:

                    best_error = error

                    best_command = command


            # ------------------------------------------------
            # Apply selected steering command
            # ------------------------------------------------

            current_steering = (
                best_command
            )


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


        # Update heading

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


        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        time_history.append(
            time
        )

        x_history.append(
            x
        )

        y_history.append(
            y
        )

        steering_history.append(
            current_steering
        )


        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

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


    max_steering = np.max(
        np.abs(
            steering_array
        )
    )


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


    return {

        "landing_error":
            landing_error,

        "flight_time":
            time,

        "landing_x":
            x,

        "landing_y":
            y,

        "wind_x":
            wind_x,

        "wind_y":
            wind_y,

        "wind_speed":
            wind_speed,

        "wind_direction":
            wind_direction_deg,

        "max_steering":
            max_steering,

        "average_steering":
            average_steering,

        "steering_reversals":
            steering_reversals
    }


# ============================================================
# MONTE CARLO SETTINGS
# ============================================================

num_simulations = 200

minimum_wind_speed = 0.0
maximum_wind_speed = 7.0


# ============================================================
# RANDOM SEED
# ============================================================

# Fixed seed makes the results repeatable.

np.random.seed(42)


# ============================================================
# GENERATE RANDOM WIND CONDITIONS
# ============================================================

wind_speeds = np.random.uniform(
    minimum_wind_speed,
    maximum_wind_speed,
    num_simulations
)

wind_directions = np.random.uniform(
    0.0,
    360.0,
    num_simulations
)


# ============================================================
# PRINT PARAMETERS
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V8.7")
print("MONTE CARLO WIND ROBUSTNESS ANALYSIS")
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

print("MONTE CARLO SETTINGS")

print(
    "Number of simulations:",
    num_simulations
)

print(
    "Wind speed range:",
    minimum_wind_speed,
    "-",
    maximum_wind_speed,
    "m/s"
)

print(
    "Wind direction range:",
    "0 - 360 degrees"
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
# RUN MONTE CARLO SIMULATIONS
# ============================================================

results = []


for i in range(
    num_simulations
):


    wind_speed = (
        wind_speeds[i]
    )

    wind_direction = (
        wind_directions[i]
    )


    result = run_simulation(
        wind_speed,
        wind_direction
    )


    results.append(
        result
    )


    # Progress display

    if (
        (i + 1) % 10 == 0
        or
        i == 0
        or
        i == num_simulations - 1
    ):

        print(
            f"Simulation "
            f"{i + 1:3d}/"
            f"{num_simulations} | "
            f"Wind = "
            f"{wind_speed:5.2f} m/s | "
            f"Direction = "
            f"{wind_direction:6.1f}° | "
            f"Error = "
            f"{result['landing_error']:7.2f} m"
        )


# ============================================================
# EXTRACT RESULTS
# ============================================================

landing_errors = np.array(
    [
        r["landing_error"]
        for r in results
    ]
)


simulation_wind_speeds = np.array(
    [
        r["wind_speed"]
        for r in results
    ]
)


simulation_wind_directions = np.array(
    [
        r["wind_direction"]
        for r in results
    ]
)


landing_x_values = np.array(
    [
        r["landing_x"]
        for r in results
    ]
)


landing_y_values = np.array(
    [
        r["landing_y"]
        for r in results
    ]
)


average_steerings = np.array(
    [
        r["average_steering"]
        for r in results
    ]
)


steering_reversals = np.array(
    [
        r["steering_reversals"]
        for r in results
    ]
)


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

mean_error = np.mean(
    landing_errors
)

median_error = np.median(
    landing_errors
)

minimum_error = np.min(
    landing_errors
)

maximum_error = np.max(
    landing_errors
)

std_error = np.std(
    landing_errors
)


# ============================================================
# PERCENTAGE WITHIN ERROR LIMITS
# ============================================================

within_5m = (
    np.sum(
        landing_errors <= 5.0
    )
    /
    num_simulations
    *
    100.0
)


within_10m = (
    np.sum(
        landing_errors <= 10.0
    )
    /
    num_simulations
    *
    100.0
)


within_20m = (
    np.sum(
        landing_errors <= 20.0
    )
    /
    num_simulations
    *
    100.0
)


within_50m = (
    np.sum(
        landing_errors <= 50.0
    )
    /
    num_simulations
    *
    100.0
)


# ============================================================
# AVERAGE CONTROLLER METRICS
# ============================================================

mean_average_steering = np.mean(
    average_steerings
)

mean_reversals = np.mean(
    steering_reversals
)


# ============================================================
# BEST CASE
# ============================================================

best_index = np.argmin(
    landing_errors
)

best_result = results[
    best_index
]


# ============================================================
# WORST CASE
# ============================================================

worst_index = np.argmax(
    landing_errors
)

worst_result = results[
    worst_index
]


# ============================================================
# MONTE CARLO RESULTS
# ============================================================

print()
print()
print("============================================================")
print("V8.7 MONTE CARLO ROBUSTNESS RESULTS")
print("============================================================")

print(
    "Number of simulations:",
    num_simulations
)

print()

print(
    "Mean landing error:",
    f"{mean_error:.3f}",
    "m"
)

print(
    "Median landing error:",
    f"{median_error:.3f}",
    "m"
)

print(
    "Standard deviation:",
    f"{std_error:.3f}",
    "m"
)

print(
    "Minimum landing error:",
    f"{minimum_error:.3f}",
    "m"
)

print(
    "Maximum landing error:",
    f"{maximum_error:.3f}",
    "m"
)

print("----------------------------------------")

print(
    "Landing within 5 m:",
    f"{within_5m:.2f}",
    "%"
)

print(
    "Landing within 10 m:",
    f"{within_10m:.2f}",
    "%"
)

print(
    "Landing within 20 m:",
    f"{within_20m:.2f}",
    "%"
)

print(
    "Landing within 50 m:",
    f"{within_50m:.2f}",
    "%"
)

print("----------------------------------------")

print(
    "Mean absolute steering:",
    f"{mean_average_steering:.3f}"
)

print(
    "Mean steering reversals:",
    f"{mean_reversals:.2f}"
)

print("============================================================")


# ============================================================
# BEST CASE
# ============================================================

print()
print("========================================")
print("BEST CASE")
print("========================================")

print(
    "Wind speed:",
    f"{best_result['wind_speed']:.3f}",
    "m/s"
)

print(
    "Wind direction:",
    f"{best_result['wind_direction']:.2f}",
    "degrees"
)

print(
    "Wind X:",
    f"{best_result['wind_x']:.3f}",
    "m/s"
)

print(
    "Wind Y:",
    f"{best_result['wind_y']:.3f}",
    "m/s"
)

print(
    "Landing X:",
    f"{best_result['landing_x']:.3f}",
    "m"
)

print(
    "Landing Y:",
    f"{best_result['landing_y']:.3f}",
    "m"
)

print(
    "Landing error:",
    f"{best_result['landing_error']:.3f}",
    "m"
)

print("========================================")


# ============================================================
# WORST CASE
# ============================================================

print()
print("========================================")
print("WORST CASE")
print("========================================")

print(
    "Wind speed:",
    f"{worst_result['wind_speed']:.3f}",
    "m/s"
)

print(
    "Wind direction:",
    f"{worst_result['wind_direction']:.2f}",
    "degrees"
)

print(
    "Wind X:",
    f"{worst_result['wind_x']:.3f}",
    "m/s"
)

print(
    "Wind Y:",
    f"{worst_result['wind_y']:.3f}",
    "m/s"
)

print(
    "Landing X:",
    f"{worst_result['landing_x']:.3f}",
    "m"
)

print(
    "Landing Y:",
    f"{worst_result['landing_y']:.3f}",
    "m"
)

print(
    "Landing error:",
    f"{worst_result['landing_error']:.3f}",
    "m"
)

print("========================================")


# ============================================================
# PLOT 1:
# LANDING ERROR DISTRIBUTION
# ============================================================

plt.figure()

plt.hist(
    landing_errors,
    bins=20
)

plt.xlabel(
    "Landing Error (m)"
)

plt.ylabel(
    "Number of Simulations"
)

plt.title(
    "V8.7 Landing Error Distribution"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 2:
# WIND SPEED VS LANDING ERROR
# ============================================================

plt.figure()

plt.scatter(
    simulation_wind_speeds,
    landing_errors
)

plt.xlabel(
    "Wind Speed (m/s)"
)

plt.ylabel(
    "Landing Error (m)"
)

plt.title(
    "Landing Error vs Wind Speed"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 3:
# WIND DIRECTION VS LANDING ERROR
# ============================================================

plt.figure()

plt.scatter(
    simulation_wind_directions,
    landing_errors
)

plt.xlabel(
    "Wind Direction (degrees)"
)

plt.ylabel(
    "Landing Error (m)"
)

plt.title(
    "Landing Error vs Wind Direction"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 4:
# LANDING POSITIONS
# ============================================================

plt.figure()

plt.scatter(
    landing_x_values,
    landing_y_values,
    label="Monte Carlo landings"
)

plt.scatter(
    target_x,
    target_y,
    marker="x",
    s=100,
    label="Target"
)

plt.xlabel(
    "Landing X (m)"
)

plt.ylabel(
    "Landing Y (m)"
)

plt.title(
    "V8.7 Monte Carlo Landing Dispersion"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 5:
# STEERING VS LANDING ERROR
# ============================================================

plt.figure()

plt.scatter(
    average_steerings,
    landing_errors
)

plt.xlabel(
    "Average Absolute Steering"
)

plt.ylabel(
    "Landing Error (m)"
)

plt.title(
    "Landing Error vs Average Steering"
)

plt.grid()

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("V8.7 SIMULATION COMPLETE")
print("========================================")

print(
    "Monte Carlo simulations:",
    num_simulations
)

print(
    "Wind speed range:",
    minimum_wind_speed,
    "-",
    maximum_wind_speed,
    "m/s"
)

print(
    "Mean landing error:",
    f"{mean_error:.3f}",
    "m"
)

print(
    "Median landing error:",
    f"{median_error:.3f}",
    "m"
)

print(
    "Worst-case landing error:",
    f"{maximum_error:.3f}",
    "m"
)

print(
    "Landing success within 10 m:",
    f"{within_10m:.2f}",
    "%"
)

print(
    "Landing success within 20 m:",
    f"{within_20m:.2f}",
    "%"
)

print("========================================")