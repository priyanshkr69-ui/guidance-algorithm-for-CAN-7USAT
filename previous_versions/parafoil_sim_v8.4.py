import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V8.4
# SHORT-HORIZON / RECEDING-HORIZON GUIDANCE
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

# Guidance updates every 2 seconds
guidance_interval = 2.0

# Short prediction horizon
prediction_horizon = 10.0

# Prediction time step
prediction_dt = 0.2

# Candidate steering commands
candidate_commands = np.linspace(
    -1.0,
    1.0,
    21
)


# ============================================================
# COST FUNCTION WEIGHTS
# ============================================================

# Weight for distance to target
distance_weight = 1.0

# Weight for heading alignment
heading_weight = 20.0

# Weight for steering effort
steering_weight = 2.0


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
# FUNCTION: ANGLE DIFFERENCE
# ============================================================

def angle_difference(
    angle1,
    angle2
):

    difference = (
        angle1 -
        angle2 +
        np.pi
    ) % (
        2 * np.pi
    ) - np.pi

    return difference


# ============================================================
# FUNCTION: PREDICT SHORT-HORIZON
# ============================================================

def predict_short_horizon(
    current_x,
    current_y,
    current_heading,
    steering_command,
    wind_x,
    wind_y
):

    px = current_x
    py = current_y

    pheading = current_heading

    prediction_time = 0.0


    while prediction_time < prediction_horizon:

        # ----------------------------------------------------
        # Turn rate
        # ----------------------------------------------------

        turn_rate = (
            max_turn_rate *
            steering_command
        )


        # ----------------------------------------------------
        # Update heading
        # ----------------------------------------------------

        pheading += (
            turn_rate *
            prediction_dt
        )


        # Keep heading between -pi and +pi

        pheading = (
            pheading + np.pi
        ) % (
            2 * np.pi
        ) - np.pi


        # ----------------------------------------------------
        # Air velocity
        # ----------------------------------------------------

        vx_air = (
            horizontal_air_velocity *
            np.cos(pheading)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(pheading)
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

        px += (
            vx_ground *
            prediction_dt
        )

        py += (
            vy_ground *
            prediction_dt
        )


        prediction_time += prediction_dt


    return (
        px,
        py,
        pheading
    )


# ============================================================
# FUNCTION: RUN ONE SIMULATION
# ============================================================

def run_simulation(
    wind_x,
    wind_y=0.0
):

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

    distance_history = []

    predicted_x_history = []
    predicted_y_history = []


    # --------------------------------------------------------
    # MAIN SIMULATION LOOP
    # --------------------------------------------------------

    while altitude > 0:


        # ====================================================
        # GUIDANCE UPDATE
        # ====================================================

        if time >= next_guidance_update:


            best_command = 0.0

            best_cost = float("inf")

            best_predicted_x = x

            best_predicted_y = y


            # ------------------------------------------------
            # Current desired direction to target
            # ------------------------------------------------

            dx_target = (
                target_x -
                x
            )

            dy_target = (
                target_y -
                y
            )


            desired_heading = np.arctan2(
                dy_target,
                dx_target
            )


            # =================================================
            # TEST EVERY CANDIDATE COMMAND
            # =================================================

            for command in candidate_commands:


                (
                    predicted_x,
                    predicted_y,
                    predicted_heading
                ) = predict_short_horizon(
                    x,
                    y,
                    heading,
                    command,
                    wind_x,
                    wind_y
                )


                # ------------------------------------------------
                # Predicted distance to target
                # ------------------------------------------------

                predicted_distance = np.sqrt(
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


                # ------------------------------------------------
                # Predicted heading error
                # ------------------------------------------------

                predicted_heading_error = abs(
                    angle_difference(
                        predicted_heading,
                        desired_heading
                    )
                )


                # ------------------------------------------------
                # Steering effort
                # ------------------------------------------------

                steering_effort = (
                    command ** 2
                )


                # ------------------------------------------------
                # Total cost
                # ------------------------------------------------

                cost = (

                    distance_weight *
                    predicted_distance

                    +

                    heading_weight *
                    predicted_heading_error

                    +

                    steering_weight *
                    steering_effort

                )


                # ------------------------------------------------
                # Keep best command
                # ------------------------------------------------

                if cost < best_cost:

                    best_cost = cost

                    best_command = command

                    best_predicted_x = predicted_x

                    best_predicted_y = predicted_y


            # =================================================
            # APPLY COMMAND
            # =================================================

            current_steering = best_command

            predicted_x_current = (
                best_predicted_x
            )

            predicted_y_current = (
                best_predicted_y
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
        # STORE DATA
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

        distance_history.append(
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

        predicted_x_history.append(
            predicted_x_current
        )

        predicted_y_history.append(
            predicted_y_current
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

        "max_steering":
            max_steering,

        "average_steering":
            average_steering,

        "steering_reversals":
            steering_reversals,

        "time_history":
            time_history,

        "x_history":
            x_history,

        "y_history":
            y_history,

        "altitude_history":
            altitude_history,

        "heading_history":
            heading_history,

        "steering_history":
            steering_history,

        "distance_history":
            distance_history,

        "predicted_x_history":
            predicted_x_history,

        "predicted_y_history":
            predicted_y_history
    }


# ============================================================
# PRINT PARAMETERS
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V8.4")
print("SHORT-HORIZON GUIDANCE")
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
    "Prediction horizon:",
    prediction_horizon,
    "s"
)

print(
    "Candidate commands:",
    len(candidate_commands)
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
# RUN WIND TESTS
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


    results.append(
        result
    )


    print(
        "Landing X:",
        result["landing_x"],
        "m"
    )

    print(
        "Landing Y:",
        result["landing_y"],
        "m"
    )

    print(
        "Landing error:",
        result["landing_error"],
        "m"
    )

    print(
        "Flight time:",
        result["flight_time"],
        "s"
    )

    print(
        "Average steering:",
        result["average_steering"]
    )

    print(
        "Steering reversals:",
        result["steering_reversals"]
    )


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

print()
print()

print("============================================================")
print("V8.4 WIND SENSITIVITY RESULTS")
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

    print(
        f"{wind:<15.1f}"
        f"{result['landing_error']:<20.3f}"
        f"{result['flight_time']:<18.2f}"
        f"{result['average_steering']:<16.3f}"
        f"{result['steering_reversals']:<12}"
    )


print(
    "============================================================"
)


# ============================================================
# LANDING ERROR VS WIND
# ============================================================

landing_errors = [

    result["landing_error"]

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
    "V8.4 Landing Error vs Wind Speed"
)

plt.grid()

plt.show()


# ============================================================
# AVERAGE STEERING VS WIND
# ============================================================

average_steerings = [

    result["average_steering"]

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
    "V8.4 Average Steering vs Wind Speed"
)

plt.grid()

plt.show()


# ============================================================
# PLOT: TRAJECTORIES
# ============================================================

plt.figure()

for wind, result in zip(
    wind_values,
    results
):

    plt.plot(
        result["x_history"],
        result["y_history"],
        label=f"{wind:.0f} m/s"
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
    "V8.4 Parafoil Trajectories Under Different Winds"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT: STEERING FOR 4 m/s WIND
# ============================================================

wind_index = wind_values.index(4.0)

result_4ms = results[wind_index]


plt.figure()

plt.step(
    result_4ms["time_history"],
    result_4ms["steering_history"],
    where="post"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Steering Command"
)

plt.title(
    "V8.4 Steering Command — 4 m/s Wind"
)

plt.grid()

plt.show()


# ============================================================
# PLOT: DISTANCE TO TARGET FOR 4 m/s WIND
# ============================================================

plt.figure()

plt.plot(
    result_4ms["time_history"],
    result_4ms["distance_history"]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Distance to Target (m)"
)

plt.title(
    "V8.4 Distance to Target — 4 m/s Wind"
)

plt.grid()

plt.show()