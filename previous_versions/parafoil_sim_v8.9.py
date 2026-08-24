import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V8.9
# REACHABILITY AND FEASIBILITY ANALYSIS
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
# FEASIBILITY PARAMETERS
# ============================================================

# Wind speeds to test

wind_speeds = np.arange(
    0.0,
    8.0,
    1.0
)


# Wind directions

wind_directions = np.arange(
    0.0,
    360.0,
    45.0
)


# Distance tolerance for declaring
# target practically reachable

reachability_tolerance = 20.0


# ============================================================
# FUNCTION:
# GET WIND COMPONENTS
# ============================================================

def wind_components(
    wind_speed,
    wind_direction_deg
):

    direction = np.radians(
        wind_direction_deg
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
# FUNCTION:
# PREDICT LANDING FOR ONE COMMAND
# ============================================================

def predict_landing(
    current_x,
    current_y,
    current_altitude,
    current_heading,
    steering_command,
    wind_x,
    wind_y
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

        # ----------------------------------------------------
        # Steering produces turn rate
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
        ) % (2 * np.pi) - np.pi

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

    return px, py


# ============================================================
# FUNCTION:
# CALCULATE REACHABLE LANDING POINTS
# ============================================================

def calculate_reachable_points(
    wind_x,
    wind_y
):

    reachable_points = []

    for command in candidate_commands:

        px, py = predict_landing(
            initial_x,
            initial_y,
            initial_altitude,
            initial_heading,
            command,
            wind_x,
            wind_y
        )

        error = np.sqrt(
            (px - target_x)**2 +
            (py - target_y)**2
        )

        reachable_points.append(
            (
                command,
                px,
                py,
                error
            )
        )

    return reachable_points


# ============================================================
# FUNCTION:
# ANALYZE ONE WIND CONDITION
# ============================================================

def analyze_wind_condition(
    wind_speed,
    wind_direction
):

    wind_x, wind_y = wind_components(
        wind_speed,
        wind_direction
    )

    points = calculate_reachable_points(
        wind_x,
        wind_y
    )

    # --------------------------------------------------------
    # Find closest reachable point
    # --------------------------------------------------------

    best_point = min(
        points,
        key=lambda p: p[3]
    )

    best_command = best_point[0]
    best_x = best_point[1]
    best_y = best_point[2]
    minimum_error = best_point[3]

    # --------------------------------------------------------
    # Check target reachability
    # --------------------------------------------------------

    target_reachable = (
        minimum_error <=
        reachability_tolerance
    )

    return {
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "wind_x": wind_x,
        "wind_y": wind_y,
        "points": points,
        "best_command": best_command,
        "best_x": best_x,
        "best_y": best_y,
        "minimum_error": minimum_error,
        "reachable": target_reachable
    }


# ============================================================
# PRINT PARAFOIL PARAMETERS
# ============================================================

print()
print("========================================")
print("VIRTUAL PARAFOIL V8.9")
print("REACHABILITY AND FEASIBILITY ANALYSIS")
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

print("INITIAL CONDITIONS")

print(
    "Altitude:",
    initial_altitude,
    "m"
)

print(
    "Initial position:",
    initial_x,
    ",",
    initial_y,
    "m"
)

print(
    "Initial heading:",
    np.degrees(initial_heading),
    "degrees"
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

print("REACHABILITY SETTINGS")

print(
    "Maximum turn rate:",
    np.degrees(max_turn_rate),
    "deg/s"
)

print(
    "Candidate steering commands:",
    len(candidate_commands)
)

print(
    "Reachability tolerance:",
    reachability_tolerance,
    "m"
)

print("----------------------------------------")

print(
    "Wind speeds tested:",
    wind_speeds
)

print(
    "Wind directions tested:",
    wind_directions
)

print("========================================")


# ============================================================
# REFERENCE CASE
# ============================================================

reference_wind_speed = 3.0
reference_wind_direction = 0.0

print()
print("========================================")
print("REFERENCE WIND CASE")
print("========================================")

reference_result = analyze_wind_condition(
    reference_wind_speed,
    reference_wind_direction
)

print(
    "Wind speed:",
    reference_result["wind_speed"],
    "m/s"
)

print(
    "Wind direction:",
    reference_result["wind_direction"],
    "degrees"
)

print(
    "Wind X:",
    reference_result["wind_x"],
    "m/s"
)

print(
    "Wind Y:",
    reference_result["wind_y"],
    "m/s"
)

print("----------------------------------------")

print(
    "Best steering command:",
    reference_result["best_command"]
)

print(
    "Closest reachable X:",
    reference_result["best_x"],
    "m"
)

print(
    "Closest reachable Y:",
    reference_result["best_y"],
    "m"
)

print(
    "Minimum achievable landing error:",
    reference_result["minimum_error"],
    "m"
)

print(
    "Target reachable:",
    "YES"
    if reference_result["reachable"]
    else "NO"
)

print("========================================")


# ============================================================
# PRINT REACHABLE POINTS FOR REFERENCE CASE
# ============================================================

print()
print("========================================")
print("REFERENCE CASE REACHABLE LANDING POINTS")
print("========================================")

print(
    f"{'Steering':<12}"
    f"{'Landing X':<15}"
    f"{'Landing Y':<15}"
    f"{'Error (m)':<15}"
)

print("----------------------------------------")

for point in reference_result["points"]:

    command, px, py, error = point

    print(
        f"{command:<12.2f}"
        f"{px:<15.2f}"
        f"{py:<15.2f}"
        f"{error:<15.2f}"
    )

print("========================================")


# ============================================================
# WIND DIRECTION ANALYSIS
# ============================================================

direction_results = []

print()
print("========================================")
print("WIND DIRECTION FEASIBILITY")
print(
    "Wind speed =",
    reference_wind_speed,
    "m/s"
)
print("========================================")

print(
    f"{'Direction':<15}"
    f"{'Min Error':<18}"
    f"{'Best Command':<18}"
    f"{'Reachable':<15}"
)

print("----------------------------------------")

for direction in wind_directions:

    result = analyze_wind_condition(
        reference_wind_speed,
        direction
    )

    direction_results.append(
        result
    )

    print(
        f"{direction:<15.1f}"
        f"{result['minimum_error']:<18.3f}"
        f"{result['best_command']:<18.2f}"
        f"{'YES' if result['reachable'] else 'NO':<15}"
    )

print("========================================")


# ============================================================
# WIND SPEED ANALYSIS
#
# For each wind speed, test all wind directions
# ============================================================

speed_results = []

print()
print("========================================")
print("WIND SPEED FEASIBILITY ANALYSIS")
print("========================================")

print(
    f"{'Wind Speed':<15}"
    f"{'Best Error':<18}"
    f"{'Worst Error':<18}"
    f"{'Reachable Dir.':<20}"
    f"{'Feasible':<15}"
)

print("----------------------------------------")


for speed in wind_speeds:

    results_for_speed = []

    for direction in wind_directions:

        result = analyze_wind_condition(
            speed,
            direction
        )

        results_for_speed.append(
            result
        )

    errors = [
        r["minimum_error"]
        for r in results_for_speed
    ]

    best_error = min(errors)

    worst_error = max(errors)

    reachable_directions = sum(
        r["reachable"]
        for r in results_for_speed
    )

    # Robust feasibility means
    # every tested wind direction
    # keeps the target within tolerance

    fully_feasible = (
        worst_error <=
        reachability_tolerance
    )

    speed_results.append(
        {
            "speed": speed,
            "best_error": best_error,
            "worst_error": worst_error,
            "reachable_directions":
                reachable_directions,
            "fully_feasible":
                fully_feasible,
            "results":
                results_for_speed
        }
    )

    print(
        f"{speed:<15.1f}"
        f"{best_error:<18.3f}"
        f"{worst_error:<18.3f}"
        f"{reachable_directions:<20}"
        f"{'YES' if fully_feasible else 'NO':<15}"
    )


print("========================================")


# ============================================================
# FIND MAXIMUM ROBUSTLY FEASIBLE WIND
# ============================================================

fully_feasible_speeds = [
    r["speed"]
    for r in speed_results
    if r["fully_feasible"]
]


if len(fully_feasible_speeds) > 0:

    maximum_feasible_wind = max(
        fully_feasible_speeds
    )

else:

    maximum_feasible_wind = None


# ============================================================
# FIND BEST AND WORST REFERENCE DIRECTIONS
# ============================================================

best_direction_result = min(
    direction_results,
    key=lambda r:
    r["minimum_error"]
)

worst_direction_result = max(
    direction_results,
    key=lambda r:
    r["minimum_error"]
)


# ============================================================
# FINAL FEASIBILITY REPORT
# ============================================================

print()
print("========================================")
print("V8.9 FEASIBILITY REPORT")
print("========================================")

print(
    "Reference wind speed:",
    reference_wind_speed,
    "m/s"
)

print("----------------------------------------")

print(
    "Best wind direction:",
    best_direction_result["wind_direction"],
    "degrees"
)

print(
    "Best minimum landing error:",
    best_direction_result["minimum_error"],
    "m"
)

print("----------------------------------------")

print(
    "Worst wind direction:",
    worst_direction_result["wind_direction"],
    "degrees"
)

print(
    "Worst minimum landing error:",
    worst_direction_result["minimum_error"],
    "m"
)

print("----------------------------------------")

if maximum_feasible_wind is not None:

    print(
        "Maximum fully feasible wind:",
        maximum_feasible_wind,
        "m/s"
    )

    print(
        "Definition:",
        "All tested directions remain within",
        reachability_tolerance,
        "m"
    )

else:

    print(
        "No wind speed in the tested range",
        "was fully feasible for all directions."
    )

print("========================================")


# ============================================================
# PLOT 1:
# REACHABLE LANDING POINTS
# REFERENCE WIND
# ============================================================

plt.figure()

points = reference_result["points"]

reachable_x = [
    p[1]
    for p in points
]

reachable_y = [
    p[2]
    for p in points
]

plt.plot(
    reachable_x,
    reachable_y,
    marker="o",
    label="Reachable landing points"
)

plt.scatter(
    target_x,
    target_y,
    s=100,
    label="Target"
)

plt.scatter(
    reference_result["best_x"],
    reference_result["best_y"],
    s=100,
    label="Closest reachable point"
)

plt.scatter(
    initial_x,
    initial_y,
    s=100,
    label="Deployment"
)

plt.xlabel(
    "X Position (m)"
)

plt.ylabel(
    "Y Position (m)"
)

plt.title(
    "V8.9 Reachable Landing Points"
)

plt.axis("equal")

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 2:
# LANDING ERROR VS STEERING COMMAND
# ============================================================

plt.figure()

commands = [
    p[0]
    for p in points
]

errors = [
    p[3]
    for p in points
]

plt.plot(
    commands,
    errors,
    marker="o"
)

plt.axhline(
    reachability_tolerance,
    linestyle="--",
    label="Reachability tolerance"
)

plt.xlabel(
    "Steering Command"
)

plt.ylabel(
    "Predicted Landing Error (m)"
)

plt.title(
    "Landing Error vs Steering Command"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 3:
# MINIMUM ERROR VS WIND DIRECTION
# ============================================================

directions = [
    r["wind_direction"]
    for r in direction_results
]

direction_errors = [
    r["minimum_error"]
    for r in direction_results
]

plt.figure()

plt.plot(
    directions,
    direction_errors,
    marker="o"
)

plt.axhline(
    reachability_tolerance,
    linestyle="--",
    label="Reachability tolerance"
)

plt.xlabel(
    "Wind Direction (degrees)"
)

plt.ylabel(
    "Minimum Achievable Landing Error (m)"
)

plt.title(
    "V8.9 Reachability vs Wind Direction"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 4:
# BEST AND WORST ERROR VS WIND SPEED
# ============================================================

speeds = [
    r["speed"]
    for r in speed_results
]

best_errors = [
    r["best_error"]
    for r in speed_results
]

worst_errors = [
    r["worst_error"]
    for r in speed_results
]

plt.figure()

plt.plot(
    speeds,
    best_errors,
    marker="o",
    label="Best direction"
)

plt.plot(
    speeds,
    worst_errors,
    marker="o",
    label="Worst direction"
)

plt.axhline(
    reachability_tolerance,
    linestyle="--",
    label="Reachability tolerance"
)

plt.xlabel(
    "Wind Speed (m/s)"
)

plt.ylabel(
    "Minimum Achievable Landing Error (m)"
)

plt.title(
    "V8.9 Wind Speed Feasibility"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 5:
# WIND FEASIBILITY MAP
# ============================================================

feasibility_matrix = []

for speed in wind_speeds:

    row = []

    for direction in wind_directions:

        result = analyze_wind_condition(
            speed,
            direction
        )

        row.append(
            1
            if result["reachable"]
            else 0
        )

    feasibility_matrix.append(
        row
    )


feasibility_matrix = np.array(
    feasibility_matrix
)


plt.figure()

plt.imshow(
    feasibility_matrix,
    aspect="auto",
    origin="lower",
    extent=[
        wind_directions[0] - 22.5,
        wind_directions[-1] + 22.5,
        wind_speeds[0] - 0.5,
        wind_speeds[-1] + 0.5
    ]
)

plt.colorbar(
    label="1 = Reachable, 0 = Not Reachable"
)

plt.xlabel(
    "Wind Direction (degrees)"
)

plt.ylabel(
    "Wind Speed (m/s)"
)

plt.title(
    "V8.9 Parafoil Reachability Map"
)

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("V8.9 SIMULATION COMPLETE")
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
    "Reference wind:",
    reference_wind_speed,
    "m/s"
)

print(
    "Reference wind direction:",
    reference_wind_direction,
    "degrees"
)

print(
    "Reference minimum landing error:",
    reference_result["minimum_error"],
    "m"
)

print("----------------------------------------")

if maximum_feasible_wind is not None:

    print(
        "Maximum fully feasible wind:",
        maximum_feasible_wind,
        "m/s"
    )

else:

    print(
        "Maximum fully feasible wind:",
        "Not found in tested range"
    )

print("----------------------------------------")

print(
    "Best tested wind direction:",
    best_direction_result["wind_direction"],
    "degrees"
)

print(
    "Worst tested wind direction:",
    worst_direction_result["wind_direction"],
    "degrees"
)

print("----------------------------------------")

print(
    "IMPORTANT:"
)

print(
    "Reachability depends on parafoil",
    "performance, altitude, wind and",
    "available steering authority."
)

print(
    "A target outside the reachable",
    "landing envelope cannot be achieved",
    "by guidance alone."
)

print("========================================")