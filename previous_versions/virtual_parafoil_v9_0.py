import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V9.0
# REACHABILITY-AWARE GUIDANCE & TRAJECTORY PLANNER
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
# INITIAL CONDITIONS
# ============================================================

initial_altitude = 600.0

initial_x = 0.0
initial_y = 0.0

initial_heading = np.radians(0.0)


# ============================================================
# TARGET
# ============================================================

target_x = 500.0
target_y = 200.0

reachability_tolerance = 20.0


# ============================================================
# WIND
# ============================================================

# Reference wind used for detailed simulation
reference_wind_speed = 3.0
reference_wind_direction = 0.0


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
print("VIRTUAL PARAFOIL V9.0")
print("REACHABILITY-AWARE GUIDANCE")
print("========================================")

print("Area:",
      area,
      "m^2")

print("Mass:",
      mass,
      "kg")

print("CL:",
      CL)

print("CD:",
      CD)

print("Airspeed:",
      airspeed,
      "m/s")

print("Horizontal air velocity:",
      horizontal_air_velocity,
      "m/s")

print("Vertical descent velocity:",
      vertical_velocity,
      "m/s")

print("Glide ratio:",
      glide_ratio)

print("----------------------------------------")

print("TARGET")

print("Target X:",
      target_x,
      "m")

print("Target Y:",
      target_y,
      "m")

print("Reachability tolerance:",
      reachability_tolerance,
      "m")

print("----------------------------------------")

print("GUIDANCE")

print("Guidance interval:",
      guidance_interval,
      "s")

print("Candidate commands:",
      len(candidate_commands))

print("Maximum turn rate:",
      np.degrees(max_turn_rate),
      "deg/s")

print()
print("Adaptive prediction horizon:")
print("Altitude > 400 m  -> 20 s")
print("Altitude 200-400 m -> 15 s")
print("Altitude 100-200 m -> 10 s")
print("Altitude < 100 m  -> 5 s")

print("========================================")


# ============================================================
# WIND COMPONENTS
# ============================================================

def calculate_wind_components(
    wind_speed,
    wind_direction_degrees
):

    direction = np.radians(
        wind_direction_degrees
    )

    wind_x = (
        wind_speed *
        np.cos(direction)
    )

    wind_y = (
        wind_speed *
        np.sin(direction)
    )

    return wind_x, wind_y


# ============================================================
# PREDICT TRAJECTORY
# ============================================================

def predict_trajectory(
    current_x,
    current_y,
    current_altitude,
    current_heading,
    steering_command,
    wind_x,
    wind_y,
    prediction_horizon
):

    prediction_dt = 0.5

    remaining_time = (
        current_altitude /
        vertical_velocity
    )

    prediction_time = min(
        prediction_horizon,
        remaining_time
    )

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

    return px, py, pheading


# ============================================================
# PREDICT FULL LANDING POINT
# ============================================================

def predict_full_landing(
    current_x,
    current_y,
    current_altitude,
    current_heading,
    steering_command,
    wind_x,
    wind_y
):

    prediction_dt = 0.5

    remaining_time = (
        current_altitude /
        vertical_velocity
    )

    steps = max(
        1,
        int(
            remaining_time /
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


# ============================================================
# BUILD REACHABLE ENVELOPE
# ============================================================

def calculate_reachable_envelope(
    wind_x,
    wind_y
):

    landing_points = []

    steering_values = []

    for command in candidate_commands:

        landing_x, landing_y = (
            predict_full_landing(
                initial_x,
                initial_y,
                initial_altitude,
                initial_heading,
                command,
                wind_x,
                wind_y
            )
        )

        landing_points.append(
            (
                landing_x,
                landing_y
            )
        )

        steering_values.append(
            command
        )

    return (
        np.array(landing_points),
        np.array(steering_values)
    )


# ============================================================
# DISTANCE TO TARGET
# ============================================================

def distance_to_target(x, y):

    return np.sqrt(
        (x - target_x) ** 2 +
        (y - target_y) ** 2
    )


# ============================================================
# REACHABILITY ANALYSIS
# ============================================================

def analyze_reachability(
    landing_points
):

    errors = []

    for point in landing_points:

        error = distance_to_target(
            point[0],
            point[1]
        )

        errors.append(error)

    errors = np.array(errors)

    best_index = np.argmin(
        errors
    )

    minimum_error = errors[
        best_index
    ]

    best_point = landing_points[
        best_index
    ]

    target_reachable = (
        minimum_error <=
        reachability_tolerance
    )

    return (
        target_reachable,
        minimum_error,
        best_point,
        errors
    )


# ============================================================
# REACHABILITY-AWARE COMMAND SELECTION
# ============================================================

def select_guidance_command(
    x,
    y,
    altitude,
    heading,
    wind_x,
    wind_y
):

    prediction_horizon = (
        get_prediction_horizon(
            altitude
        )
    )

    best_command = 0.0

    best_cost = float("inf")

    best_prediction = (
        x,
        y
    )

    # --------------------------------------------------------
    # Distance from current state to target
    # --------------------------------------------------------

    current_target_distance = (
        distance_to_target(
            x,
            y
        )
    )

    # --------------------------------------------------------
    # Test every steering command
    # --------------------------------------------------------

    for command in candidate_commands:

        px, py, pheading = (
            predict_trajectory(
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

        predicted_target_distance = (
            distance_to_target(
                px,
                py
            )
        )

        # ----------------------------------------------------
        # Heading toward target
        # ----------------------------------------------------

        target_bearing = np.arctan2(
            target_y - py,
            target_x - px
        )

        heading_error = abs(
            (
                target_bearing -
                pheading +
                np.pi
            ) %
            (2 * np.pi)
            - np.pi
        )

        heading_cost = (
            heading_error /
            np.pi
        )

        # ----------------------------------------------------
        # Progress cost
        # ----------------------------------------------------

        progress_cost = (
            predicted_target_distance /
            max(
                current_target_distance,
                1.0
            )
        )

        # ----------------------------------------------------
        # Steering effort penalty
        # ----------------------------------------------------

        steering_cost = (
            0.05 *
            abs(command)
        )

        # ----------------------------------------------------
        # Total cost
        # ----------------------------------------------------

        total_cost = (
            0.65 *
            progress_cost
            +
            0.25 *
            heading_cost
            +
            steering_cost
        )

        if total_cost < best_cost:

            best_cost = total_cost

            best_command = command

            best_prediction = (
                px,
                py
            )

    return (
        best_command,
        best_prediction,
        prediction_horizon
    )


# ============================================================
# RUN FULL GUIDANCE SIMULATION
# ============================================================

def run_guidance_simulation(
    wind_x,
    wind_y
):

    altitude = initial_altitude

    x = initial_x
    y = initial_y

    heading = initial_heading

    dt = 0.1

    time = 0.0

    next_guidance_update = 0.0

    current_steering = 0.0

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    time_history = []

    x_history = []
    y_history = []

    altitude_history = []

    heading_history = []

    steering_history = []

    predicted_x_history = []
    predicted_y_history = []

    prediction_horizon_history = []

    distance_history = []

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    while altitude > 0:

        # ----------------------------------------------------
        # Guidance update
        # ----------------------------------------------------

        if time >= next_guidance_update:

            (
                current_steering,
                prediction,
                horizon
            ) = select_guidance_command(
                x,
                y,
                altitude,
                heading,
                wind_x,
                wind_y
            )

            predicted_x_current = (
                prediction[0]
            )

            predicted_y_current = (
                prediction[1]
            )

            prediction_horizon_current = (
                horizon
            )

            next_guidance_update = (
                time +
                guidance_interval
            )

        # ----------------------------------------------------
        # Parafoil dynamics
        # ----------------------------------------------------

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

        predicted_x_history.append(
            predicted_x_current
        )

        predicted_y_history.append(
            predicted_y_current
        )

        prediction_horizon_history.append(
            prediction_horizon_current
        )

        distance_history.append(
            distance_to_target(
                x,
                y
            )
        )

        time += dt

    # --------------------------------------------------------
    # Final landing error
    # --------------------------------------------------------

    landing_error = distance_to_target(
        x,
        y
    )

    # --------------------------------------------------------
    # Controller statistics
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

    return {
        "landing_x": x,
        "landing_y": y,
        "landing_error": landing_error,
        "flight_time": time,
        "average_steering": average_steering,
        "steering_reversals": steering_reversals,
        "time": time_history,
        "x": x_history,
        "y": y_history,
        "altitude": altitude_history,
        "heading": heading_history,
        "steering": steering_history,
        "predicted_x": predicted_x_history,
        "predicted_y": predicted_y_history,
        "prediction_horizon": prediction_horizon_history,
        "distance": distance_history
    }


# ============================================================
# REACHABILITY ENVELOPE FOR REFERENCE WIND
# ============================================================

(
    reference_wind_x,
    reference_wind_y
) = calculate_wind_components(
    reference_wind_speed,
    reference_wind_direction
)

print()
print("========================================")
print("REFERENCE WIND")
print("========================================")

print(
    "Wind speed:",
    reference_wind_speed,
    "m/s"
)

print(
    "Wind direction:",
    reference_wind_direction,
    "degrees"
)

print(
    "Wind X:",
    reference_wind_x,
    "m/s"
)

print(
    "Wind Y:",
    reference_wind_y,
    "m/s"
)


# ============================================================
# CALCULATE REACHABLE LANDING POINTS
# ============================================================

(
    reachable_points,
    reachable_commands
) = calculate_reachable_envelope(
    reference_wind_x,
    reference_wind_y
)

(
    target_reachable,
    minimum_reachable_error,
    closest_reachable_point,
    reachable_errors
) = analyze_reachability(
    reachable_points
)


# ============================================================
# REACHABILITY RESULTS
# ============================================================

print()
print("========================================")
print("REACHABILITY ANALYSIS")
print("========================================")

print(
    "Target:",
    target_x,
    ",",
    target_y,
    "m"
)

print(
    "Reachability tolerance:",
    reachability_tolerance,
    "m"
)

print(
    "Minimum predicted landing error:",
    minimum_reachable_error,
    "m"
)

print(
    "Closest reachable landing point:",
    closest_reachable_point[0],
    ",",
    closest_reachable_point[1],
    "m"
)

if target_reachable:

    print()
    print("TARGET STATUS: REACHABLE")

else:

    print()
    print("TARGET STATUS: NOT REACHABLE")

print("========================================")


# ============================================================
# RUN ACTUAL GUIDANCE SIMULATION
# ============================================================

print()
print("----------------------------------------")
print("RUNNING REACHABILITY-AWARE GUIDANCE")
print("----------------------------------------")

simulation = run_guidance_simulation(
    reference_wind_x,
    reference_wind_y
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("========================================")
print("V9.0 FINAL RESULTS")
print("========================================")

print(
    "Landing X:",
    simulation["landing_x"],
    "m"
)

print(
    "Landing Y:",
    simulation["landing_y"],
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
    "Final landing error:",
    simulation["landing_error"],
    "m"
)

print(
    "Flight time:",
    simulation["flight_time"],
    "s"
)

print(
    "Average steering:",
    simulation["average_steering"]
)

print(
    "Steering reversals:",
    simulation["steering_reversals"]
)

print("========================================")


# ============================================================
# COMPARISON: IDEAL REACHABLE POINT VS ACTUAL
# ============================================================

actual_point = np.array(
    [
        simulation["landing_x"],
        simulation["landing_y"]
    ]
)

actual_to_closest = np.linalg.norm(
    actual_point -
    closest_reachable_point
)

print()
print("========================================")
print("REACHABILITY COMPARISON")
print("========================================")

print(
    "Closest predicted reachable point:",
    closest_reachable_point[0],
    ",",
    closest_reachable_point[1]
)

print(
    "Actual landing point:",
    actual_point[0],
    ",",
    actual_point[1]
)

print(
    "Distance between predicted closest",
    "point and actual landing:",
    actual_to_closest,
    "m"
)

print("========================================")


# ============================================================
# PLOT 1: REACHABLE LANDING ENVELOPE
# ============================================================

plt.figure()

plt.scatter(
    reachable_points[:, 0],
    reachable_points[:, 1],
    label="Reachable landing points"
)

plt.scatter(
    target_x,
    target_y,
    s=100,
    label="Target"
)

plt.scatter(
    closest_reachable_point[0],
    closest_reachable_point[1],
    s=100,
    label="Closest reachable point"
)

plt.scatter(
    simulation["landing_x"],
    simulation["landing_y"],
    s=100,
    label="Actual landing"
)

plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "V9.0 Reachable Landing Envelope"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 2: ACTUAL TRAJECTORY
# ============================================================

plt.figure()

plt.plot(
    simulation["x"],
    simulation["y"],
    label="Actual trajectory"
)

plt.scatter(
    initial_x,
    initial_y,
    s=100,
    label="Deployment"
)

plt.scatter(
    target_x,
    target_y,
    s=100,
    label="Target"
)

plt.scatter(
    closest_reachable_point[0],
    closest_reachable_point[1],
    s=100,
    label="Closest reachable point"
)

plt.scatter(
    simulation["landing_x"],
    simulation["landing_y"],
    s=100,
    label="Landing"
)

plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "V9.0 Parafoil Trajectory"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 3: DISTANCE TO TARGET
# ============================================================

plt.figure()

plt.plot(
    simulation["time"],
    simulation["distance"]
)

plt.axhline(
    reachability_tolerance,
    linestyle="--",
    label="Reachability tolerance"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Distance to Target (m)"
)

plt.title(
    "Distance to Target"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 4: STEERING COMMAND
# ============================================================

plt.figure()

plt.step(
    simulation["time"],
    simulation["steering"],
    where="post"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Steering Command"
)

plt.title(
    "V9.0 Steering Command"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 5: ALTITUDE
# ============================================================

plt.figure()

plt.plot(
    simulation["time"],
    simulation["altitude"]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Altitude (m)"
)

plt.title(
    "Parafoil Altitude During Guidance"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 6: HEADING
# ============================================================

plt.figure()

plt.plot(
    simulation["time"],
    simulation["heading"]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Heading (degrees)"
)

plt.title(
    "Parafoil Heading"
)

plt.grid()

plt.show()


# ============================================================
# FINAL DIAGNOSTIC
# ============================================================

print()
print("========================================")
print("V9.0 DIAGNOSTIC SUMMARY")
print("========================================")

if target_reachable:

    print(
        "The target is physically reachable",
        "under the reference wind condition."
    )

else:

    print(
        "The target is NOT physically reachable",
        "within the specified tolerance."
    )

print()
print(
    "The guidance system therefore attempts",
    "to minimize landing error while respecting"
)

print(
    "the available steering authority and",
    "current wind condition."
)

print()
print(
    "Minimum reachable error:",
    minimum_reachable_error,
    "m"
)

print(
    "Actual final error:",
    simulation["landing_error"],
    "m"
)

print()
print("========================================")
print("V9.0 SIMULATION COMPLETE")
print("========================================")