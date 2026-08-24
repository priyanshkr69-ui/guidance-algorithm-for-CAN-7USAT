import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V8.8
# WIND-COMPENSATED ADAPTIVE GUIDANCE
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
# WIND-COMPENSATION PARAMETERS
# ============================================================

# Weight given to wind-compensated heading error
heading_weight = 0.35

# Weight given to predicted landing position error
position_weight = 0.65


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
# ANGLE WRAP FUNCTION
# ============================================================

def wrap_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


# ============================================================
# WIND-COMPENSATED DESIRED HEADING
# ============================================================

def calculate_desired_heading(
    current_x,
    current_y,
    wind_x,
    wind_y
):

    # --------------------------------------------------------
    # Vector from current position to target
    # --------------------------------------------------------

    dx = target_x - current_x
    dy = target_y - current_y

    distance = np.sqrt(
        dx**2 +
        dy**2
    )

    # Avoid division problems
    if distance < 1e-6:

        return 0.0


    # --------------------------------------------------------
    # Target bearing
    # --------------------------------------------------------

    target_bearing = np.arctan2(
        dy,
        dx
    )


    # --------------------------------------------------------
    # Wind vector
    # --------------------------------------------------------

    wind_speed = np.sqrt(
        wind_x**2 +
        wind_y**2
    )


    # --------------------------------------------------------
    # If there is almost no wind,
    # simply point toward target
    # --------------------------------------------------------

    if wind_speed < 1e-6:

        return target_bearing


    # --------------------------------------------------------
    # Determine heading candidates
    #
    # We search over possible headings and find the heading
    # whose ground velocity points closest to the target.
    # --------------------------------------------------------

    heading_candidates = np.linspace(
        -np.pi,
        np.pi,
        361
    )

    best_heading = target_bearing

    best_error = float("inf")


    for heading in heading_candidates:

        vx_air = (
            horizontal_air_velocity *
            np.cos(heading)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(heading)
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


        ground_speed = np.sqrt(
            vx_ground**2 +
            vy_ground**2
        )


        if ground_speed < 1e-6:

            continue


        ground_heading = np.arctan2(
            vy_ground,
            vx_ground
        )


        heading_error = abs(
            wrap_angle(
                ground_heading -
                target_bearing
            )
        )


        if heading_error < best_error:

            best_error = heading_error

            best_heading = heading


    return best_heading


# ============================================================
# FUNCTION: RUN ONE SIMULATION
# ============================================================

def run_simulation(
    wind_speed,
    wind_direction_deg
):

    # ========================================================
    # WIND COMPONENTS
    # ========================================================

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


    # ========================================================
    # INITIAL CONDITIONS
    # ========================================================

    altitude = 600.0

    x = 0.0
    y = 0.0

    heading = np.radians(0.0)

    dt = 0.1

    time = 0.0

    next_guidance_update = 0.0

    current_steering = 0.0


    # ========================================================
    # HISTORY
    # ========================================================

    time_history = []

    x_history = []
    y_history = []

    heading_history = []

    steering_history = []

    distance_history = []

    desired_heading_history = []

    predicted_error_history = []


    # ========================================================
    # MAIN SIMULATION LOOP
    # ========================================================

    while altitude > 0:

        # ====================================================
        # GUIDANCE UPDATE
        # ====================================================

        if time >= next_guidance_update:


            # ------------------------------------------------
            # Adaptive horizon
            # ------------------------------------------------

            prediction_horizon = (
                get_prediction_horizon(
                    altitude
                )
            )


            # ------------------------------------------------
            # Wind-compensated desired heading
            # ------------------------------------------------

            desired_heading = (
                calculate_desired_heading(
                    x,
                    y,
                    wind_x,
                    wind_y
                )
            )


            # ------------------------------------------------
            # Candidate command search
            # ------------------------------------------------

            best_command = 0.0

            best_cost = float("inf")

            best_predicted_x = x
            best_predicted_y = y


            for command in candidate_commands:


                # ============================================
                # SHORT-HORIZON PREDICTION
                # ============================================

                px = x
                py = y

                pheading = heading

                prediction_dt = 0.5

                steps = max(
                    1,
                    int(
                        prediction_horizon /
                        prediction_dt
                    )
                )


                for _ in range(steps):


                    # ----------------------------------------
                    # Candidate turn rate
                    # ----------------------------------------

                    turn_rate = (
                        max_turn_rate *
                        command
                    )


                    # ----------------------------------------
                    # Heading update
                    # ----------------------------------------

                    pheading += (
                        turn_rate *
                        prediction_dt
                    )

                    pheading = wrap_angle(
                        pheading
                    )


                    # ----------------------------------------
                    # Air velocity
                    # ----------------------------------------

                    vx_air = (
                        horizontal_air_velocity *
                        np.cos(pheading)
                    )

                    vy_air = (
                        horizontal_air_velocity *
                        np.sin(pheading)
                    )


                    # ----------------------------------------
                    # Ground velocity
                    # ----------------------------------------

                    vx_ground = (
                        vx_air +
                        wind_x
                    )

                    vy_ground = (
                        vy_air +
                        wind_y
                    )


                    # ----------------------------------------
                    # Position update
                    # ----------------------------------------

                    px += (
                        vx_ground *
                        prediction_dt
                    )

                    py += (
                        vy_ground *
                        prediction_dt
                    )


                # ============================================
                # POSITION ERROR
                # ============================================

                position_error = np.sqrt(
                    (
                        px -
                        target_x
                    )**2
                    +
                    (
                        py -
                        target_y
                    )**2
                )


                # ============================================
                # HEADING ERROR
                # ============================================

                command_heading_error = abs(
                    wrap_angle(
                        pheading -
                        desired_heading
                    )
                )


                # ============================================
                # NORMALIZE HEADING ERROR
                # ============================================

                normalized_heading_error = (
                    command_heading_error /
                    np.pi
                )


                # ============================================
                # COMBINED COST
                # ============================================

                cost = (
                    position_weight *
                    position_error
                    +
                    heading_weight *
                    normalized_heading_error *
                    100.0
                )


                # ============================================
                # SELECT BEST COMMAND
                # ============================================

                if cost < best_cost:

                    best_cost = cost

                    best_command = command

                    best_predicted_x = px

                    best_predicted_y = py


            # ------------------------------------------------
            # Apply selected command
            # ------------------------------------------------

            current_steering = (
                best_command
            )


            # ------------------------------------------------
            # Predicted error
            # ------------------------------------------------

            predicted_error = np.sqrt(
                (
                    best_predicted_x -
                    target_x
                )**2
                +
                (
                    best_predicted_y -
                    target_y
                )**2
            )


            # ------------------------------------------------
            # Next guidance update
            # ------------------------------------------------

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


        # ----------------------------------------------------
        # Update heading
        # ----------------------------------------------------

        heading += (
            turn_rate *
            dt
        )

        heading = wrap_angle(
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
                )**2
                +
                (
                    y -
                    target_y
                )**2
            )
        )

        desired_heading_history.append(
            np.degrees(
                desired_heading
            )
        )

        predicted_error_history.append(
            predicted_error
        )


        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

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

        "heading_history":
            heading_history,

        "steering_history":
            steering_history,

        "distance_history":
            distance_history,

        "desired_heading_history":
            desired_heading_history,

        "predicted_error_history":
            predicted_error_history
    }


# ============================================================
# PRINT PARAMETERS
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V8.8")
print("WIND-COMPENSATED ADAPTIVE GUIDANCE")
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

print(
    "Position cost weight:",
    position_weight
)

print(
    "Heading cost weight:",
    heading_weight
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
# WIND TESTS
# ============================================================

wind_speed = 3.0

wind_directions = [
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


# ============================================================
# RUN WIND-DIRECTION TESTS
# ============================================================

for direction in wind_directions:

    print()
    print("----------------------------------------")

    print(
        "Running simulation for wind direction:",
        direction,
        "degrees"
    )


    result = run_simulation(
        wind_speed,
        direction
    )


    results.append(
        result
    )


    wind_direction_rad = np.radians(
        direction
    )


    wind_x = (
        wind_speed *
        np.cos(wind_direction_rad)
    )

    wind_y = (
        wind_speed *
        np.sin(wind_direction_rad)
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
# RESULTS TABLE
# ============================================================

print()
print()
print(
    "============================================================"
)

print(
    "V8.8 WIND-COMPENSATED RESULTS"
)

print(
    "============================================================"
)

print(
    f"{'Direction':<15}"
    f"{'Wind X':<15}"
    f"{'Wind Y':<15}"
    f"{'Landing Error':<18}"
    f"{'Avg Steering':<16}"
    f"{'Reversals':<12}"
)

print(
    "------------------------------------------------------------"
)


for direction, result in zip(
    wind_directions,
    results
):

    direction_rad = np.radians(
        direction
    )

    wx = (
        wind_speed *
        np.cos(direction_rad)
    )

    wy = (
        wind_speed *
        np.sin(direction_rad)
    )


    print(
        f"{direction:<15.1f}"
        f"{wx:<15.3f}"
        f"{wy:<15.3f}"
        f"{result['landing_error']:<18.3f}"
        f"{result['average_steering']:<16.3f}"
        f"{result['steering_reversals']:<12}"
    )


print(
    "============================================================"
)


# ============================================================
# BEST AND WORST CASE
# ============================================================

errors = np.array([
    result["landing_error"]
    for result in results
])


best_index = np.argmin(
    errors
)

worst_index = np.argmax(
    errors
)


# ============================================================
# BEST CASE
# ============================================================

print()
print(
    "========================================"
)

print(
    "BEST WIND DIRECTION"
)

print(
    "Direction:",
    wind_directions[best_index],
    "degrees"
)

print(
    "Landing error:",
    errors[best_index],
    "m"
)

print(
    "========================================"
)


# ============================================================
# WORST CASE
# ============================================================

print()
print(
    "========================================"
)

print(
    "WORST WIND DIRECTION"
)

print(
    "Direction:",
    wind_directions[worst_index],
    "degrees"
)

print(
    "Landing error:",
    errors[worst_index],
    "m"
)

print(
    "========================================"
)


# ============================================================
# LANDING ERROR VS WIND DIRECTION
# ============================================================

plt.figure()

plt.plot(
    wind_directions,
    errors,
    marker="o"
)

plt.xlabel(
    "Wind Direction (degrees)"
)

plt.ylabel(
    "Landing Error (m)"
)

plt.title(
    "V8.8 Landing Error vs Wind Direction"
)

plt.grid()

plt.show()


# ============================================================
# DETAILED REFERENCE CASE
# ============================================================

reference_direction = 0.0

reference_result = results[
    wind_directions.index(
        reference_direction
    )
]


# ============================================================
# REFERENCE TRAJECTORY
# ============================================================

plt.figure()

plt.plot(
    reference_result["x_history"],
    reference_result["y_history"],
    label="Actual trajectory"
)

plt.scatter(
    0.0,
    0.0,
    label="Deployment"
)

plt.scatter(
    target_x,
    target_y,
    label="Target"
)

plt.scatter(
    reference_result["landing_x"],
    reference_result["landing_y"],
    label="Landing"
)

plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "V8.8 Wind-Compensated Trajectory"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# REFERENCE STEERING
# ============================================================

plt.figure()

plt.step(
    reference_result["time_history"],
    reference_result["steering_history"],
    where="post"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Steering Command"
)

plt.title(
    "V8.8 Wind-Compensated Steering Command"
)

plt.grid()

plt.show()


# ============================================================
# REFERENCE DISTANCE TO TARGET
# ============================================================

plt.figure()

plt.plot(
    reference_result["time_history"],
    reference_result["distance_history"]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Distance to Target (m)"
)

plt.title(
    "V8.8 Distance to Target"
)

plt.grid()

plt.show()


# ============================================================
# REFERENCE HEADING
# ============================================================

plt.figure()

plt.plot(
    reference_result["time_history"],
    reference_result["heading_history"],
    label="Actual heading"
)

plt.plot(
    reference_result["time_history"],
    reference_result["desired_heading_history"],
    label="Desired heading"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Heading (degrees)"
)

plt.title(
    "V8.8 Actual vs Desired Heading"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# REFERENCE PREDICTION ERROR
# ============================================================

plt.figure()

plt.plot(
    reference_result["time_history"],
    reference_result["predicted_error_history"]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Predicted Error (m)"
)

plt.title(
    "V8.8 Predicted Landing Error"
)

plt.grid()

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print(
    "========================================"
)

print(
    "V8.8 SIMULATION COMPLETE"
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
    "Directions tested:",
    len(wind_directions)
)

print(
    "Best landing error:",
    errors[best_index],
    "m"
)

print(
    "Worst landing error:",
    errors[worst_index],
    "m"
)

print(
    "Reference wind direction:",
    reference_direction,
    "degrees"
)

print(
    "Reference landing position:",
    reference_result["landing_x"],
    ",",
    reference_result["landing_y"],
    "m"
)

print(
    "Reference landing error:",
    reference_result["landing_error"],
    "m"
)

print(
    "========================================"
)