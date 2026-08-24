import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V9.2
# DYNAMIC REACHABILITY + MONTE CARLO VALIDATION
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V9.2")
print("DYNAMIC REACHABILITY + MONTE CARLO")
print("VALIDATION")
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
# DYNAMIC REACHABILITY / BEAM SEARCH
# ============================================================

beam_width = 80

search_interval = 5.0

prediction_dt = 0.5


# ============================================================
# MONTE CARLO SETTINGS
# ============================================================

num_simulations = 200

wind_min = 0.0
wind_max = 7.0

random_seed = 42

rng = np.random.default_rng(random_seed)


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
# PRINT PARAMETERS
# ============================================================

print("Area:", area, "m^2")
print("Mass:", mass, "kg")
print("CL:", CL)
print("CD:", CD)

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

print("----------------------------------------")

print("MONTE CARLO SETTINGS")

print(
    "Number of simulations:",
    num_simulations
)

print(
    "Wind speed range:",
    wind_min,
    "-",
    wind_max,
    "m/s"
)

print(
    "Wind direction range: 0 - 360 degrees"
)

print("----------------------------------------")

print("BEAM SEARCH")

print(
    "Beam width:",
    beam_width
)

print(
    "Search interval:",
    search_interval,
    "s"
)

print(
    "Candidate steering commands:",
    len(candidate_commands)
)

print(
    "Maximum turn rate:",
    np.degrees(max_turn_rate),
    "deg/s"
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
# STATE PROPAGATION
# ============================================================
#
# RETURNS:
# x
# y
# altitude
# heading
#
# This is intentionally kept consistent throughout V9.2.
# ============================================================

def propagate_state(
    x,
    y,
    altitude,
    heading,
    steering_command,
    duration,
    wind_x,
    wind_y
):

    steps = max(
        1,
        int(
            np.ceil(
                duration /
                prediction_dt
            )
        )
    )

    dt = duration / steps

    px = x
    py = y
    paltitude = altitude
    pheading = heading

    for _ in range(steps):

        # Steering command -> turn rate

        turn_rate = (
            max_turn_rate *
            steering_command
        )

        # Update heading

        pheading += (
            turn_rate *
            dt
        )

        pheading = (
            pheading + np.pi
        ) % (
            2.0 * np.pi
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

        # Position

        px += (
            vx_ground *
            dt
        )

        py += (
            vy_ground *
            dt
        )

        # Altitude

        paltitude -= (
            vertical_velocity *
            dt
        )

        if paltitude <= 0.0:

            paltitude = 0.0

            break

    return (
        px,
        py,
        paltitude,
        pheading
    )


# ============================================================
# PREDICT LANDING FOR A CONSTANT STEERING COMMAND
# ============================================================

def predict_landing(
    x,
    y,
    altitude,
    heading,
    steering_command,
    wind_x,
    wind_y
):

    if altitude <= 0.0:

        return (
            x,
            y,
            heading
        )

    remaining_time = (
        altitude /
        vertical_velocity
    )

    (
        px,
        py,
        _,
        pheading
    ) = propagate_state(
        x,
        y,
        altitude,
        heading,
        steering_command,
        remaining_time,
        wind_x,
        wind_y
    )

    return (
        px,
        py,
        pheading
    )


# ============================================================
# STATE COST
# ============================================================

def calculate_state_cost(
    x,
    y,
    heading,
    altitude,
    wind_x,
    wind_y
):

    remaining_time = (
        altitude /
        vertical_velocity
    )

    if remaining_time <= 0.0:

        return np.sqrt(
            (x - target_x) ** 2 +
            (y - target_y) ** 2
        )

    # Direction from current position to target

    dx = target_x - x
    dy = target_y - y

    target_bearing = np.arctan2(
        dy,
        dx
    )

    # Heading error

    heading_error = (
        target_bearing -
        heading
    )

    heading_error = (
        heading_error + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    # Position error

    position_error = np.sqrt(
        dx ** 2 +
        dy ** 2
    )

    # Normalize heading error

    normalized_heading_error = (
        abs(heading_error) /
        np.pi
    )

    # Cost

    position_weight = 0.75
    heading_weight = 0.25

    cost = (
        position_weight *
        position_error
        +
        heading_weight *
        normalized_heading_error *
        100.0
    )

    return cost


# ============================================================
# BEAM NODE
# ============================================================

class BeamNode:

    def __init__(
        self,
        x,
        y,
        altitude,
        heading,
        commands,
        cost
    ):

        self.x = x
        self.y = y
        self.altitude = altitude
        self.heading = heading

        self.commands = commands.copy()

        self.cost = cost


# ============================================================
# BUILD DYNAMIC REACHABLE ENVELOPE
# ============================================================

def build_reachable_envelope(
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

    stages = max(
        1,
        int(
            np.ceil(
                prediction_horizon /
                search_interval
            )
        )
    )

    initial_node = BeamNode(
        x,
        y,
        altitude,
        heading,
        [],
        0.0
    )

    beam = [
        initial_node
    ]

    # --------------------------------------------------------
    # Beam search
    # --------------------------------------------------------

    for stage in range(stages):

        new_beam = []

        for node in beam:

            for command in candidate_commands:

                (
                    nx,
                    ny,
                    naltitude,
                    nheading
                ) = propagate_state(
                    node.x,
                    node.y,
                    node.altitude,
                    node.heading,
                    command,
                    search_interval,
                    wind_x,
                    wind_y
                )

                cost = calculate_state_cost(
                    nx,
                    ny,
                    nheading,
                    naltitude,
                    wind_x,
                    wind_y
                )

                new_commands = (
                    node.commands +
                    [command]
                )

                new_node = BeamNode(
                    nx,
                    ny,
                    naltitude,
                    nheading,
                    new_commands,
                    cost
                )

                new_beam.append(
                    new_node
                )

        if len(new_beam) == 0:

            break

        # Keep only best beam states

        new_beam.sort(
            key=lambda node:
            node.cost
        )

        beam = new_beam[
            :beam_width
        ]

    return beam


# ============================================================
# SELECT BEST GUIDANCE COMMAND
# ============================================================

def select_guidance_command(
    x,
    y,
    altitude,
    heading,
    wind_x,
    wind_y
):

    # --------------------------------------------------------
    # If very close to ground, point toward target
    # --------------------------------------------------------

    if altitude < 30.0:

        dx = target_x - x
        dy = target_y - y

        target_bearing = np.arctan2(
            dy,
            dx
        )

        heading_error = (
            target_bearing -
            heading
        )

        heading_error = (
            heading_error + np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi

        command = (
            heading_error /
            max_turn_rate
        )

        command = np.clip(
            command,
            -1.0,
            1.0
        )

        return (
            command,
            np.sqrt(
                dx ** 2 +
                dy ** 2
            )
        )

    # --------------------------------------------------------
    # Build dynamic reachable states
    # --------------------------------------------------------

    beam = build_reachable_envelope(
        x,
        y,
        altitude,
        heading,
        wind_x,
        wind_y
    )

    if len(beam) == 0:

        return (
            0.0,
            float("inf")
        )

    # --------------------------------------------------------
    # Find best predicted state
    # --------------------------------------------------------

    best_node = None
    best_error = float("inf")

    for node in beam:

        (
            px,
            py,
            _
        ) = predict_landing(
            node.x,
            node.y,
            node.altitude,
            node.heading,
            0.0,
            wind_x,
            wind_y
        )

        error = np.sqrt(
            (px - target_x) ** 2 +
            (py - target_y) ** 2
        )

        if error < best_error:

            best_error = error

            best_node = node

    # --------------------------------------------------------
    # First command of best sequence
    # --------------------------------------------------------

    if (
        best_node is None
        or
        len(best_node.commands) == 0
    ):

        return (
            0.0,
            best_error
        )

    command = (
        best_node.commands[0]
    )

    return (
        command,
        best_error
    )


# ============================================================
# INITIAL REACHABILITY TEST
# ============================================================

def analyze_reachability(
    wind_x,
    wind_y
):

    beam = build_reachable_envelope(
        0.0,
        0.0,
        600.0,
        0.0,
        wind_x,
        wind_y
    )

    minimum_error = float("inf")

    closest_x = None
    closest_y = None

    for node in beam:

        (
            px,
            py,
            _
        ) = predict_landing(
            node.x,
            node.y,
            node.altitude,
            node.heading,
            0.0,
            wind_x,
            wind_y
        )

        error = np.sqrt(
            (px - target_x) ** 2 +
            (py - target_y) ** 2
        )

        if error < minimum_error:

            minimum_error = error

            closest_x = px
            closest_y = py

    reachable = (
        minimum_error <=
        reachability_tolerance
    )

    return (
        reachable,
        minimum_error,
        closest_x,
        closest_y
    )


# ============================================================
# RUN CLOSED-LOOP SIMULATION
# ============================================================

def run_simulation(
    wind_x,
    wind_y
):

    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------

    altitude = 600.0

    x = 0.0
    y = 0.0

    heading = 0.0

    dt = 0.1

    time = 0.0

    next_guidance_update = 0.0

    current_steering = 0.0

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    x_history = []
    y_history = []
    altitude_history = []
    heading_history = []
    steering_history = []
    time_history = []

    predicted_error_history = []

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while altitude > 0.0:

        # ----------------------------------------------------
        # Guidance update
        # ----------------------------------------------------

        if time >= next_guidance_update:

            (
                new_command,
                predicted_error
            ) = select_guidance_command(
                x,
                y,
                altitude,
                heading,
                wind_x,
                wind_y
            )

            current_steering = (
                new_command
            )

            predicted_error_history.append(
                predicted_error
            )

            next_guidance_update = (
                time +
                guidance_interval
            )

        # ----------------------------------------------------
        # Actual parafoil dynamics
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
        ) % (
            2.0 * np.pi
        ) - np.pi

        # Air velocity

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

        # Position

        x += (
            vx_ground *
            dt
        )

        y += (
            vy_ground *
            dt
        )

        # Altitude

        altitude -= (
            vertical_velocity *
            dt
        )

        if altitude < 0.0:

            altitude = 0.0

        # Store history

        x_history.append(x)
        y_history.append(y)
        altitude_history.append(
            altitude
        )

        heading_history.append(
            heading
        )

        steering_history.append(
            current_steering
        )

        time_history.append(
            time
        )

        time += dt

    # --------------------------------------------------------
    # Landing error
    # --------------------------------------------------------

    landing_error = np.sqrt(
        (x - target_x) ** 2 +
        (y - target_y) ** 2
    )

    # --------------------------------------------------------
    # Steering analysis
    # --------------------------------------------------------

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
            steering_array[i] != 0.0
            and
            steering_array[i - 1] != 0.0
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
    # Predicted minimum
    # --------------------------------------------------------

    if len(
        predicted_error_history
    ) > 0:

        minimum_predicted_error = min(
            predicted_error_history
        )

    else:

        minimum_predicted_error = float(
            "inf"
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "landing_error":
            landing_error,

        "landing_x":
            x,

        "landing_y":
            y,

        "flight_time":
            time,

        "average_steering":
            average_steering,

        "max_steering":
            max_steering,

        "steering_reversals":
            steering_reversals,

        "minimum_predicted_error":
            minimum_predicted_error,

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

        "time_history":
            time_history
    }


# ============================================================
# GENERATE MONTE CARLO WIND CONDITIONS
# ============================================================

wind_speeds = rng.uniform(
    wind_min,
    wind_max,
    num_simulations
)

wind_directions = rng.uniform(
    0.0,
    360.0,
    num_simulations
)


# ============================================================
# STORAGE
# ============================================================

landing_errors = []

predicted_errors = []

wind_speed_results = []

wind_direction_results = []

landing_x_results = []

landing_y_results = []

average_steering_results = []

reversal_results = []

reachability_results = []


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

print()
print("========================================")
print("STARTING MONTE CARLO VALIDATION")
print("========================================")

for i in range(
    num_simulations
):

    wind_speed = (
        wind_speeds[i]
    )

    wind_direction = (
        wind_directions[i]
    )

    direction_rad = np.radians(
        wind_direction
    )

    wind_x = (
        wind_speed *
        np.cos(direction_rad)
    )

    wind_y = (
        wind_speed *
        np.sin(direction_rad)
    )

    # --------------------------------------------------------
    # Reachability analysis
    # --------------------------------------------------------

    (
        reachable,
        minimum_reachable_error,
        closest_x,
        closest_y
    ) = analyze_reachability(
        wind_x,
        wind_y
    )

    # --------------------------------------------------------
    # Closed-loop simulation
    # --------------------------------------------------------

    simulation = run_simulation(
        wind_x,
        wind_y
    )

    landing_error = (
        simulation[
            "landing_error"
        ]
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    landing_errors.append(
        landing_error
    )

    predicted_errors.append(
        minimum_reachable_error
    )

    wind_speed_results.append(
        wind_speed
    )

    wind_direction_results.append(
        wind_direction
    )

    landing_x_results.append(
        simulation[
            "landing_x"
        ]
    )

    landing_y_results.append(
        simulation[
            "landing_y"
        ]
    )

    average_steering_results.append(
        simulation[
            "average_steering"
        ]
    )

    reversal_results.append(
        simulation[
            "steering_reversals"
        ]
    )

    reachability_results.append(
        reachable
    )

    # --------------------------------------------------------
    # Progress output
    # --------------------------------------------------------

    if (
        i == 0
        or
        (i + 1) % 10 == 0
        or
        i == num_simulations - 1
    ):

        print(
            f"Simulation {i + 1:3d}/"
            f"{num_simulations} | "
            f"Wind = "
            f"{wind_speed:5.2f} m/s | "
            f"Direction = "
            f"{wind_direction:6.1f}° | "
            f"Predicted = "
            f"{minimum_reachable_error:7.2f} m | "
            f"Actual = "
            f"{landing_error:7.2f} m"
        )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

landing_errors = np.array(
    landing_errors
)

predicted_errors = np.array(
    predicted_errors
)

wind_speed_results = np.array(
    wind_speed_results
)

wind_direction_results = np.array(
    wind_direction_results
)

landing_x_results = np.array(
    landing_x_results
)

landing_y_results = np.array(
    landing_y_results
)

average_steering_results = np.array(
    average_steering_results
)

reversal_results = np.array(
    reversal_results
)

reachability_results = np.array(
    reachability_results
)


# ============================================================
# STATISTICS
# ============================================================

mean_error = np.mean(
    landing_errors
)

median_error = np.median(
    landing_errors
)

std_error = np.std(
    landing_errors
)

minimum_error = np.min(
    landing_errors
)

maximum_error = np.max(
    landing_errors
)

mean_predicted_error = np.mean(
    predicted_errors
)

median_predicted_error = np.median(
    predicted_errors
)


# ============================================================
# SUCCESS RATES
# ============================================================

success_5m = (
    np.sum(
        landing_errors <= 5.0
    )
    /
    num_simulations
    *
    100.0
)

success_10m = (
    np.sum(
        landing_errors <= 10.0
    )
    /
    num_simulations
    *
    100.0
)

success_20m = (
    np.sum(
        landing_errors <= 20.0
    )
    /
    num_simulations
    *
    100.0
)

success_50m = (
    np.sum(
        landing_errors <= 50.0
    )
    /
    num_simulations
    *
    100.0
)


# ============================================================
# REACHABILITY STATISTICS
# ============================================================

reachable_count = np.sum(
    reachability_results
)

reachable_percentage = (
    reachable_count /
    num_simulations *
    100.0
)


# ============================================================
# PREDICTION QUALITY
# ============================================================

prediction_difference = (
    np.abs(
        landing_errors -
        predicted_errors
    )
)

mean_prediction_difference = np.mean(
    prediction_difference
)

median_prediction_difference = np.median(
    prediction_difference
)


# ============================================================
# STEERING STATISTICS
# ============================================================

mean_steering = np.mean(
    average_steering_results
)

mean_reversals = np.mean(
    reversal_results
)


# ============================================================
# BEST CASE
# ============================================================

best_index = np.argmin(
    landing_errors
)

best_error = (
    landing_errors[
        best_index
    ]
)

best_wind_speed = (
    wind_speed_results[
        best_index
    ]
)

best_direction = (
    wind_direction_results[
        best_index
    ]
)

best_wind_x = (
    best_wind_speed *
    np.cos(
        np.radians(
            best_direction
        )
    )
)

best_wind_y = (
    best_wind_speed *
    np.sin(
        np.radians(
            best_direction
        )
    )
)


# ============================================================
# WORST CASE
# ============================================================

worst_index = np.argmax(
    landing_errors
)

worst_error = (
    landing_errors[
        worst_index
    ]
)

worst_wind_speed = (
    wind_speed_results[
        worst_index
    ]
)

worst_direction = (
    wind_direction_results[
        worst_index
    ]
)

worst_wind_x = (
    worst_wind_speed *
    np.cos(
        np.radians(
            worst_direction
        )
    )
)

worst_wind_y = (
    worst_wind_speed *
    np.sin(
        np.radians(
            worst_direction
        )
    )
)


# ============================================================
# BEST PREDICTED REACHABLE CASE
# ============================================================

best_predicted_index = np.argmin(
    predicted_errors
)

best_predicted_error = (
    predicted_errors[
        best_predicted_index
    ]
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print()
print("============================================================")
print("V9.2 MONTE CARLO VALIDATION RESULTS")
print("============================================================")

print(
    "Number of simulations:",
    num_simulations
)

print()

print(
    f"Mean landing error: "
    f"{mean_error:.3f} m"
)

print(
    f"Median landing error: "
    f"{median_error:.3f} m"
)

print(
    f"Standard deviation: "
    f"{std_error:.3f} m"
)

print(
    f"Minimum landing error: "
    f"{minimum_error:.3f} m"
)

print(
    f"Maximum landing error: "
    f"{maximum_error:.3f} m"
)

print("----------------------------------------")

print(
    f"Landing within 5 m: "
    f"{success_5m:.2f} %"
)

print(
    f"Landing within 10 m: "
    f"{success_10m:.2f} %"
)

print(
    f"Landing within 20 m: "
    f"{success_20m:.2f} %"
)

print(
    f"Landing within 50 m: "
    f"{success_50m:.2f} %"
)

print("----------------------------------------")

print(
    f"Reachable cases: "
    f"{reachable_count}/"
    f"{num_simulations}"
)

print(
    f"Reachability percentage: "
    f"{reachable_percentage:.2f} %"
)

print("----------------------------------------")

print(
    f"Mean predicted reachable error: "
    f"{mean_predicted_error:.3f} m"
)

print(
    f"Median predicted reachable error: "
    f"{median_predicted_error:.3f} m"
)

print(
    f"Mean prediction difference: "
    f"{mean_prediction_difference:.3f} m"
)

print(
    f"Median prediction difference: "
    f"{median_prediction_difference:.3f} m"
)

print("----------------------------------------")

print(
    f"Mean absolute steering: "
    f"{mean_steering:.3f}"
)

print(
    f"Mean steering reversals: "
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
    f"Wind speed: "
    f"{best_wind_speed:.3f} m/s"
)

print(
    f"Wind direction: "
    f"{best_direction:.2f} degrees"
)

print(
    f"Wind X: "
    f"{best_wind_x:.3f} m/s"
)

print(
    f"Wind Y: "
    f"{best_wind_y:.3f} m/s"
)

print(
    f"Landing X: "
    f"{landing_x_results[best_index]:.3f} m"
)

print(
    f"Landing Y: "
    f"{landing_y_results[best_index]:.3f} m"
)

print(
    f"Landing error: "
    f"{best_error:.3f} m"
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
    f"Wind speed: "
    f"{worst_wind_speed:.3f} m/s"
)

print(
    f"Wind direction: "
    f"{worst_direction:.2f} degrees"
)

print(
    f"Wind X: "
    f"{worst_wind_x:.3f} m/s"
)

print(
    f"Wind Y: "
    f"{worst_wind_y:.3f} m/s"
)

print(
    f"Landing X: "
    f"{landing_x_results[worst_index]:.3f} m"
)

print(
    f"Landing Y: "
    f"{landing_y_results[worst_index]:.3f} m"
)

print(
    f"Landing error: "
    f"{worst_error:.3f} m"
)

print("========================================")


# ============================================================
# BEST PREDICTED REACHABLE CASE
# ============================================================

print()
print("========================================")
print("BEST PREDICTED REACHABLE CASE")
print("========================================")

print(
    f"Minimum predicted reachable error: "
    f"{best_predicted_error:.3f} m"
)

print(
    f"Wind speed: "
    f"{wind_speed_results[best_predicted_index]:.3f} m/s"
)

print(
    f"Wind direction: "
    f"{wind_direction_results[best_predicted_index]:.2f} degrees"
)

print("========================================")


# ============================================================
# PLOT 1
# LANDING ERROR DISTRIBUTION
# ============================================================

plt.figure()

plt.hist(
    landing_errors,
    bins=25
)

plt.axvline(
    5.0,
    linestyle="--",
    label="5 m"
)

plt.axvline(
    10.0,
    linestyle="--",
    label="10 m"
)

plt.axvline(
    20.0,
    linestyle="--",
    label="20 m"
)

plt.xlabel(
    "Landing Error (m)"
)

plt.ylabel(
    "Number of Simulations"
)

plt.title(
    "V9.2 Monte Carlo Landing Error Distribution"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 2
# LANDING ERROR VS WIND SPEED
# ============================================================

plt.figure()

plt.scatter(
    wind_speed_results,
    landing_errors,
    alpha=0.7
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
# PLOT 3
# LANDING ERROR VS WIND DIRECTION
# ============================================================

plt.figure()

plt.scatter(
    wind_direction_results,
    landing_errors,
    alpha=0.7
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
# PLOT 4
# LANDING POINTS
# ============================================================

plt.figure()

plt.scatter(
    landing_x_results,
    landing_y_results,
    alpha=0.6,
    label="Simulated Landings"
)

plt.scatter(
    target_x,
    target_y,
    marker="x",
    s=100,
    label="Target"
)

circle = plt.Circle(
    (
        target_x,
        target_y
    ),
    reachability_tolerance,
    fill=False,
    linestyle="--",
    label="20 m tolerance"
)

plt.gca().add_patch(
    circle
)

plt.xlabel(
    "Landing X (m)"
)

plt.ylabel(
    "Landing Y (m)"
)

plt.title(
    "V9.2 Monte Carlo Landing Distribution"
)

plt.axis("equal")

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 5
# PREDICTED VS ACTUAL ERROR
# ============================================================

plt.figure()

plt.scatter(
    predicted_errors,
    landing_errors,
    alpha=0.6
)

max_value = max(
    np.max(predicted_errors),
    np.max(landing_errors)
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--",
    label="Ideal prediction"
)

plt.xlabel(
    "Predicted Reachable Error (m)"
)

plt.ylabel(
    "Actual Landing Error (m)"
)

plt.title(
    "Reachability Prediction vs Actual Landing"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("V9.2 SIMULATION COMPLETE")
print("========================================")

print(
    "Monte Carlo simulations:",
    num_simulations
)

print(
    "Wind speed range:",
    wind_min,
    "-",
    wind_max,
    "m/s"
)

print(
    f"Mean landing error: "
    f"{mean_error:.3f} m"
)

print(
    f"Median landing error: "
    f"{median_error:.3f} m"
)

print(
    f"Worst-case landing error: "
    f"{maximum_error:.3f} m"
)

print(
    f"Landing success within 10 m: "
    f"{success_10m:.2f} %"
)

print(
    f"Landing success within 20 m: "
    f"{success_20m:.2f} %"
)

print(
    f"Predicted reachable cases: "
    f"{reachable_percentage:.2f} %"
)

print("========================================")

print()
print("NEXT DEVELOPMENT STEP:")
print("V10 -> SENSOR NOISE + EKF STATE ESTIMATION")
print()