import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V9.1
# DYNAMIC REACHABILITY ANALYSIS
#
# Corrected version
#
# State definition:
#     x
#     y
#     heading
#     altitude
#
# Beam-search reachable envelope
# + closed-loop reachability-aware guidance
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
# REFERENCE WIND
# ============================================================

reference_wind_speed = 3.0
reference_wind_direction = 0.0

reference_wind_x = (
    reference_wind_speed *
    np.cos(np.radians(reference_wind_direction))
)

reference_wind_y = (
    reference_wind_speed *
    np.sin(np.radians(reference_wind_direction))
)


# ============================================================
# GUIDANCE PARAMETERS
# ============================================================

guidance_interval = 2.0

dt = 0.1

max_turn_rate = np.radians(15.0)


# Candidate steering commands

candidate_commands = np.linspace(
    -1.0,
    1.0,
    21
)


# ============================================================
# BEAM SEARCH PARAMETERS
# ============================================================

beam_width = 80

search_interval = 5.0


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
print("VIRTUAL PARAFOIL V9.1")
print("DYNAMIC REACHABILITY ANALYSIS")
print("========================================")

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

print(
    "TARGET:",
    target_x,
    ",",
    target_y,
    "m"
)

print(
    "Tolerance:",
    reachability_tolerance,
    "m"
)

print("----------------------------------------")

print("REFERENCE WIND")

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

print("========================================")


# ============================================================
# FUNCTION: ADAPTIVE HORIZON
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
# FUNCTION: NORMALIZE ANGLE
# ============================================================

def normalize_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


# ============================================================
# FUNCTION: PROPAGATE STATE
#
# IMPORTANT:
# This function ALWAYS returns:
#
#     x
#     y
#     heading
#     altitude
#
# This fixes the V9.1 unpacking error.
# ============================================================

def propagate_state(
    state,
    steering_command,
    duration,
    wind_x,
    wind_y
):

    x = state[0]
    y = state[1]
    heading = state[2]
    altitude = state[3]

    elapsed = 0.0

    while (
        elapsed < duration
        and
        altitude > 0.0
    ):

        step = min(
            dt,
            duration - elapsed
        )

        # ----------------------------------------------------
        # TURN RATE
        # ----------------------------------------------------

        turn_rate = (
            max_turn_rate *
            steering_command
        )

        # ----------------------------------------------------
        # HEADING
        # ----------------------------------------------------

        heading += (
            turn_rate *
            step
        )

        heading = normalize_angle(
            heading
        )

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
            step
        )

        y += (
            vy_ground *
            step
        )

        # ----------------------------------------------------
        # ALTITUDE
        # ----------------------------------------------------

        altitude -= (
            vertical_velocity *
            step
        )

        elapsed += step

    return (
        x,
        y,
        heading,
        altitude
    )


# ============================================================
# FUNCTION: LANDING ERROR
# ============================================================

def landing_error(
    x,
    y
):

    return np.sqrt(
        (x - target_x) ** 2 +
        (y - target_y) ** 2
    )


# ============================================================
# FUNCTION: HEADING TO TARGET
# ============================================================

def target_heading(
    x,
    y
):

    return np.arctan2(
        target_y - y,
        target_x - x
    )


# ============================================================
# BEAM NODE
# ============================================================

class BeamNode:

    def __init__(
        self,
        state,
        commands
    ):

        self.state = state

        self.commands = commands


# ============================================================
# BEAM SEARCH
#
# Builds a set of dynamically reachable states.
# ============================================================

def build_reachable_envelope(
    wind_x,
    wind_y
):

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    initial_state = (
        initial_x,
        initial_y,
        initial_heading,
        initial_altitude
    )

    beam = [
        BeamNode(
            initial_state,
            []
        )
    ]


    # Number of search stages

    total_time = (
        initial_altitude /
        vertical_velocity
    )

    number_of_stages = int(
        np.ceil(
            total_time /
            search_interval
        )
    )


    # --------------------------------------------------------
    # BEAM SEARCH
    # --------------------------------------------------------

    for stage in range(
        number_of_stages
    ):

        new_nodes = []

        for node in beam:

            state = node.state

            # ------------------------------------------------
            # Try every steering command
            # ------------------------------------------------

            for command in candidate_commands:

                new_state = propagate_state(
                    state,
                    command,
                    search_interval,
                    wind_x,
                    wind_y
                )

                new_commands = (
                    node.commands +
                    [command]
                )

                new_nodes.append(
                    BeamNode(
                        new_state,
                        new_commands
                    )
                )


        # ----------------------------------------------------
        # Remove states that already reached ground
        # ----------------------------------------------------

        active_nodes = []

        for node in new_nodes:

            if node.state[3] > 0.0:

                active_nodes.append(node)


        # If all states have landed, stop

        if len(active_nodes) == 0:

            beam = new_nodes

            break


        # ----------------------------------------------------
        # SCORE STATES
        # ----------------------------------------------------

        scored_nodes = []

        for node in active_nodes:

            x = node.state[0]
            y = node.state[1]
            heading = node.state[2]

            position_error = landing_error(
                x,
                y
            )

            desired_heading = target_heading(
                x,
                y
            )

            heading_error = abs(
                normalize_angle(
                    desired_heading -
                    heading
                )
            )

            # Convert heading error to approximate
            # distance penalty.

            heading_penalty = (
                heading_error /
                np.pi
            ) * 50.0

            score = (
                position_error +
                heading_penalty
            )

            scored_nodes.append(
                (
                    score,
                    node
                )
            )


        # ----------------------------------------------------
        # Keep best beam_width states
        # ----------------------------------------------------

        scored_nodes.sort(
            key=lambda item: item[0]
        )

        beam = [
            item[1]
            for item in
            scored_nodes[
                :beam_width
            ]
        ]


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            stage == 0
            or
            (stage + 1) % 5 == 0
            or
            stage == number_of_stages - 1
        ):

            best_node = beam[0]

            print(
                "Search stage:",
                stage + 1,
                "/",
                number_of_stages,
                "| Beam states:",
                len(beam),
                "| Best position:",
                round(best_node.state[0], 2),
                ",",
                round(best_node.state[1], 2),
                "| Altitude:",
                round(
                    best_node.state[3],
                    2
                ),
                "m"
            )


    return beam


# ============================================================
# COMPLETE LANDING POINTS FROM BEAM
#
# Each beam state is propagated until landing using its
# complete steering sequence.
# ============================================================

def evaluate_beam_landings(
    beam,
    wind_x,
    wind_y
):

    landing_states = []

    for node in beam:

        state = node.state

        remaining_altitude = (
            state[3]
        )

        remaining_time = (
            remaining_altitude /
            vertical_velocity
        )

        # Continue using the last command

        if len(node.commands) > 0:

            final_command = (
                node.commands[-1]
            )

        else:

            final_command = 0.0


        final_state = propagate_state(
            state,
            final_command,
            remaining_time,
            wind_x,
            wind_y
        )

        error = landing_error(
            final_state[0],
            final_state[1]
        )

        landing_states.append(
            (
                error,
                final_state,
                node.commands
            )
        )


    landing_states.sort(
        key=lambda item: item[0]
    )

    return landing_states


# ============================================================
# FUNCTION: SELECT GUIDANCE COMMAND
#
# At each guidance update, test candidate commands.
# The command whose future state is closest to the best
# reachable landing prediction is selected.
# ============================================================

def select_guidance_command(
    x,
    y,
    heading,
    altitude,
    wind_x,
    wind_y
):

    current_state = (
        x,
        y,
        heading,
        altitude
    )

    horizon = get_prediction_horizon(
        altitude
    )

    best_command = 0.0

    best_score = float("inf")

    best_predicted_state = (
        x,
        y,
        heading,
        altitude
    )


    # --------------------------------------------------------
    # Try every steering command
    # --------------------------------------------------------

    for command in candidate_commands:

        predicted_state = propagate_state(
            current_state,
            command,
            horizon,
            wind_x,
            wind_y
        )

        px = predicted_state[0]
        py = predicted_state[1]
        pheading = predicted_state[2]

        # Position error

        position_error = landing_error(
            px,
            py
        )

        # Desired heading

        desired_heading = target_heading(
            px,
            py
        )

        heading_error = abs(
            normalize_angle(
                desired_heading -
                pheading
            )
        )

        heading_cost = (
            heading_error /
            np.pi
        ) * 30.0

        score = (
            position_error +
            heading_cost
        )

        if score < best_score:

            best_score = score

            best_command = command

            best_predicted_state = (
                predicted_state
            )


    return (
        best_command,
        best_predicted_state,
        best_score
    )


# ============================================================
# RUN CLOSED-LOOP SIMULATION
# ============================================================

def run_simulation(
    wind_x,
    wind_y
):

    altitude = (
        initial_altitude
    )

    x = initial_x
    y = initial_y

    heading = (
        initial_heading
    )

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

    predicted_x_history = []
    predicted_y_history = []

    prediction_error_history = []


    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while altitude > 0.0:

        # ----------------------------------------------------
        # GUIDANCE UPDATE
        # ----------------------------------------------------

        if (
            time >=
            next_guidance_update
        ):

            (
                current_steering,
                predicted_state,
                predicted_score
            ) = select_guidance_command(
                x,
                y,
                heading,
                altitude,
                wind_x,
                wind_y
            )

            predicted_x = (
                predicted_state[0]
            )

            predicted_y = (
                predicted_state[1]
            )

            predicted_error = (
                landing_error(
                    predicted_x,
                    predicted_y
                )
            )

            next_guidance_update = (
                time +
                guidance_interval
            )


        # ----------------------------------------------------
        # ACTUAL DYNAMICS
        # ----------------------------------------------------

        state = (
            x,
            y,
            heading,
            altitude
        )

        new_state = propagate_state(
            state,
            current_steering,
            dt,
            wind_x,
            wind_y
        )

        x = new_state[0]
        y = new_state[1]
        heading = new_state[2]
        altitude = new_state[3]


        # ----------------------------------------------------
        # STORE DATA
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
            np.degrees(heading)
        )

        steering_history.append(
            current_steering
        )

        predicted_x_history.append(
            predicted_x
        )

        predicted_y_history.append(
            predicted_y
        )

        prediction_error_history.append(
            predicted_error
        )


        time += dt


    # --------------------------------------------------------
    # FINAL ERROR
    # --------------------------------------------------------

    final_error = landing_error(
        x,
        y
    )


    # --------------------------------------------------------
    # CONTROLLER ANALYSIS
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
            final_error,

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

        "time":
            time_history,

        "x":
            x_history,

        "y":
            y_history,

        "altitude":
            altitude_history,

        "heading":
            heading_history,

        "steering":
            steering_history,

        "predicted_x":
            predicted_x_history,

        "predicted_y":
            predicted_y_history,

        "predicted_error":
            prediction_error_history
    }


# ============================================================
# BUILD DYNAMIC REACHABLE ENVELOPE
# ============================================================

print()
print("----------------------------------------")
print("BUILDING DYNAMIC REACHABLE ENVELOPE")
print("----------------------------------------")

beam = build_reachable_envelope(
    reference_wind_x,
    reference_wind_y
)


# ============================================================
# EVALUATE LANDING POINTS
# ============================================================

landing_states = evaluate_beam_landings(
    beam,
    reference_wind_x,
    reference_wind_y
)


# ============================================================
# FIND CLOSEST REACHABLE POINT
# ============================================================

best_reachable = (
    landing_states[0]
)


minimum_predicted_error = (
    best_reachable[0]
)

closest_state = (
    best_reachable[1]
)

best_sequence = (
    best_reachable[2]
)


# ============================================================
# REACHABILITY RESULTS
# ============================================================

print()
print("========================================")
print("V9.1 REACHABILITY RESULTS")
print("========================================")

print(
    "Number of reachable states:",
    len(landing_states)
)

print(
    "Minimum predicted landing error:",
    minimum_predicted_error,
    "m"
)

print(
    "Closest reachable landing point:",
    closest_state[0],
    ",",
    closest_state[1],
    "m"
)


if (
    minimum_predicted_error
    <=
    reachability_tolerance
):

    target_reachable = True

    print()
    print(
        "TARGET STATUS: REACHABLE"
    )

else:

    target_reachable = False

    print()
    print(
        "TARGET STATUS: NOT REACHABLE"
    )


print("========================================")


# ============================================================
# BEST PREDICTED STEERING SEQUENCE
# ============================================================

print()
print("========================================")
print("BEST PREDICTED STEERING SEQUENCE")
print("========================================")

print(
    "Number of commands:",
    len(best_sequence)
)

for i, command in enumerate(
    best_sequence
):

    print(
        f"Stage {i + 1:02d}: "
        f"Steering = {command:+.2f}"
    )


# ============================================================
# RUN CLOSED-LOOP GUIDANCE
# ============================================================

print()
print("----------------------------------------")
print("RUNNING CLOSED LOOP GUIDANCE")
print("----------------------------------------")


simulation = run_simulation(
    reference_wind_x,
    reference_wind_y
)


# ============================================================
# EXTRACT RESULTS
# ============================================================

landing_x = (
    simulation["landing_x"]
)

landing_y = (
    simulation["landing_y"]
)

final_error = (
    simulation["landing_error"]
)

flight_time = (
    simulation["flight_time"]
)

average_steering = (
    simulation["average_steering"]
)

steering_reversals = (
    simulation["steering_reversals"]
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("========================================")
print("V9.1 FINAL RESULTS")
print("========================================")

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
    final_error,
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

print("========================================")


# ============================================================
# REACHABILITY COMPARISON
# ============================================================

distance_from_reachable_point = np.sqrt(
    (
        landing_x -
        closest_state[0]
    ) ** 2
    +
    (
        landing_y -
        closest_state[1]
    ) ** 2
)


print()
print("========================================")
print("REACHABILITY COMPARISON")
print("========================================")

print(
    "Closest predicted reachable point:",
    closest_state[0],
    ",",
    closest_state[1]
)

print(
    "Actual landing point:",
    landing_x,
    ",",
    landing_y
)

print(
    "Distance between predicted closest "
    "point and actual landing:",
    distance_from_reachable_point,
    "m"
)


# ============================================================
# DIAGNOSTIC
# ============================================================

print()
print("========================================")
print("V9.1 DIAGNOSTIC SUMMARY")
print("========================================")

if target_reachable:

    print(
        "The target is dynamically reachable "
        "within the specified tolerance."
    )

else:

    print(
        "The target is NOT dynamically reachable "
        "within the specified tolerance."
    )

    print(
        "The guidance system therefore attempts "
        "to minimize landing error while respecting "
        "the available steering authority."
    )


print()
print(
    "Minimum predicted reachable error:",
    minimum_predicted_error,
    "m"
)

print(
    "Actual final error:",
    final_error,
    "m"
)

print("========================================")


# ============================================================
# PLOT 1: ACTUAL TRAJECTORY + REACHABLE LANDING POINTS
# ============================================================

reachable_x = []

reachable_y = []

for item in landing_states:

    state = item[1]

    reachable_x.append(
        state[0]
    )

    reachable_y.append(
        state[1]
    )


plt.figure()

plt.scatter(
    reachable_x,
    reachable_y,
    s=15,
    alpha=0.5,
    label="Reachable landing states"
)

plt.plot(
    simulation["x"],
    simulation["y"],
    label="Actual trajectory"
)

plt.scatter(
    initial_x,
    initial_y,
    label="Deployment"
)

plt.scatter(
    target_x,
    target_y,
    s=80,
    marker="x",
    label="Target"
)

plt.scatter(
    closest_state[0],
    closest_state[1],
    s=80,
    marker="D",
    label="Closest predicted point"
)

plt.scatter(
    landing_x,
    landing_y,
    s=80,
    marker="o",
    label="Actual landing"
)

plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "V9.1 Dynamic Reachability Envelope"
)

plt.axis(
    "equal"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 2: DISTANCE TO TARGET
# ============================================================

actual_distance = []

for x_value, y_value in zip(
    simulation["x"],
    simulation["y"]
):

    actual_distance.append(
        landing_error(
            x_value,
            y_value
        )
    )


plt.figure()

plt.plot(
    simulation["time"],
    actual_distance
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Distance to Target (m)"
)

plt.title(
    "V9.1 Actual Distance to Target"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 3: PREDICTED LANDING ERROR
# ============================================================

plt.figure()

plt.plot(
    simulation["time"],
    simulation["predicted_error"]
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Predicted Error (m)"
)

plt.title(
    "V9.1 Predicted Landing Error"
)

plt.grid()

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
    "V9.1 Closed-Loop Steering Command"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 5: HEADING
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
    "V9.1 Parafoil Heading"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 6: ALTITUDE
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
    "V9.1 Altitude Profile"
)

plt.grid()

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("========================================")
print("V9.1 SIMULATION COMPLETE")
print("========================================")

print(
    "Reference wind:",
    reference_wind_speed,
    "m/s"
)

print(
    "Reference direction:",
    reference_wind_direction,
    "degrees"
)

print(
    "Reachability tolerance:",
    reachability_tolerance,
    "m"
)

print(
    "Minimum predicted reachable error:",
    minimum_predicted_error,
    "m"
)

print(
    "Actual landing error:",
    final_error,
    "m"
)

print(
    "Actual landing position:",
    landing_x,
    ",",
    landing_y,
    "m"
)

print("========================================")