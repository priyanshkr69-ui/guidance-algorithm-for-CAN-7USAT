import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V6.1
# NUMERIC WIND-COMPENSATED GUIDANCE
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

altitude = 600.0

x = 0.0
y = 0.0

heading = np.radians(0.0)


# ============================================================
# TARGET
# ============================================================

target_x = 500.0
target_y = 200.0


# ============================================================
# CONTROLLER
# ============================================================

Kp = 1.5

max_turn_rate = np.radians(15.0)


# ============================================================
# WIND
# ============================================================

wind_x = 3.0
wind_y = 0.0


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
# PRINT
# ============================================================

print("----------------------------------------")
print("VIRTUAL PARAFOIL V6.1")
print("----------------------------------------")

print("Parafoil area:", area, "m^2")
print("Mass:", mass, "kg")
print("CL:", CL)
print("CD:", CD)

print("Airspeed:",
      airspeed, "m/s")

print("Horizontal air velocity:",
      horizontal_air_velocity, "m/s")

print("Vertical descent velocity:",
      vertical_velocity, "m/s")

print("Glide ratio:",
      glide_ratio)

print("----------------------------------------")

print("TARGET")
print("Target X:", target_x, "m")
print("Target Y:", target_y, "m")

print("----------------------------------------")

print("WIND")
print("Wind X:", wind_x, "m/s")
print("Wind Y:", wind_y, "m/s")

print("----------------------------------------")


# ============================================================
# SIMULATION
# ============================================================

dt = 0.1
time = 0.0


# ============================================================
# HISTORY
# ============================================================

time_history = []

x_history = []
y_history = []

altitude_history = []

heading_history = []

desired_heading_history = []

ground_track_history = []

steering_history = []

distance_history = []


# ============================================================
# MAIN LOOP
# ============================================================

while altitude > 0:

    # --------------------------------------------------------
    # TARGET VECTOR
    # --------------------------------------------------------

    dx = target_x - x
    dy = target_y - y

    distance_to_target = np.sqrt(
        dx**2 + dy**2
    )

    target_bearing = np.arctan2(
        dy,
        dx
    )


    # --------------------------------------------------------
    # NUMERICALLY FIND BEST HEADING
    # --------------------------------------------------------

    candidate_headings = np.linspace(
        -np.pi,
        np.pi,
        721
    )


    best_heading = candidate_headings[0]

    best_error = 1e9


    for candidate in candidate_headings:

        # Air-relative velocity

        vx_air = (
            horizontal_air_velocity *
            np.cos(candidate)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(candidate)
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


        # Ground-track direction

        ground_track = np.arctan2(
            vy_ground,
            vx_ground
        )


        # Difference between ground track
        # and target direction

        angle_error = (
            ground_track -
            target_bearing
        )


        angle_error = (
            angle_error + np.pi
        ) % (2 * np.pi) - np.pi


        # We minimize absolute angular error

        if abs(angle_error) < best_error:

            best_error = abs(angle_error)

            best_heading = candidate


    # --------------------------------------------------------
    # DESIRED HEADING
    # --------------------------------------------------------

    desired_heading = best_heading


    # --------------------------------------------------------
    # HEADING ERROR
    # --------------------------------------------------------

    heading_error = (
        desired_heading -
        heading
    )

    heading_error = (
        heading_error + np.pi
    ) % (2 * np.pi) - np.pi


    # --------------------------------------------------------
    # CONTROLLER
    # --------------------------------------------------------

    steering_command = (
        Kp * heading_error
    )

    steering_command = np.clip(
        steering_command,
        -1.0,
        1.0
    )


    # --------------------------------------------------------
    # TURN RATE
    # --------------------------------------------------------

    turn_rate = (
        max_turn_rate *
        steering_command
    )


    # --------------------------------------------------------
    # UPDATE HEADING
    # --------------------------------------------------------

    heading = (
        heading +
        turn_rate * dt
    )

    heading = (
        heading + np.pi
    ) % (2 * np.pi) - np.pi


    # --------------------------------------------------------
    # AIR VELOCITY
    # --------------------------------------------------------

    vx_air = (
        horizontal_air_velocity *
        np.cos(heading)
    )

    vy_air = (
        horizontal_air_velocity *
        np.sin(heading)
    )


    # --------------------------------------------------------
    # GROUND VELOCITY
    # --------------------------------------------------------

    vx_ground = (
        vx_air +
        wind_x
    )

    vy_ground = (
        vy_air +
        wind_y
    )


    # --------------------------------------------------------
    # GROUND TRACK
    # --------------------------------------------------------

    ground_track = np.arctan2(
        vy_ground,
        vx_ground
    )


    # --------------------------------------------------------
    # UPDATE POSITION
    # --------------------------------------------------------

    x += vx_ground * dt
    y += vy_ground * dt


    # --------------------------------------------------------
    # UPDATE ALTITUDE
    # --------------------------------------------------------

    altitude -= vertical_velocity * dt


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    time_history.append(time)

    x_history.append(x)
    y_history.append(y)

    altitude_history.append(altitude)

    heading_history.append(
        np.degrees(heading)
    )

    desired_heading_history.append(
        np.degrees(desired_heading)
    )

    ground_track_history.append(
        np.degrees(ground_track)
    )

    steering_history.append(
        steering_command
    )

    distance_history.append(
        distance_to_target
    )


    time += dt


# ============================================================
# FINAL RESULTS
# ============================================================

landing_error = np.sqrt(
    (x - target_x)**2 +
    (y - target_y)**2
)


print()
print("----------------------------------------")
print("FINAL RESULTS")
print("----------------------------------------")

print("Flight time:",
      time, "seconds")

print("Landing X:",
      x, "m")

print("Landing Y:",
      y, "m")

print("Target X:",
      target_x, "m")

print("Target Y:",
      target_y, "m")

print("Landing error:",
      landing_error, "m")

print("Final heading:",
      np.degrees(heading), "degrees")

print("----------------------------------------")


# ============================================================
# GROUND TRACK
# ============================================================

plt.figure()

plt.plot(
    x_history,
    y_history,
    label="Parafoil trajectory"
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

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")

plt.title(
    "V6.1 Wind-Compensated Ground Track"
)

plt.axis("equal")

plt.grid()

plt.legend()

plt.show()


# ============================================================
# HEADING
# ============================================================

plt.figure()

plt.plot(
    time_history,
    heading_history,
    label="Actual heading"
)

plt.plot(
    time_history,
    desired_heading_history,
    label="Desired heading"
)

plt.plot(
    time_history,
    ground_track_history,
    label="Ground track direction"
)

plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")

plt.title(
    "Heading and Ground Track"
)

plt.grid()

plt.legend()

plt.show()


# ============================================================
# STEERING
# ============================================================

plt.figure()

plt.plot(
    time_history,
    steering_history
)

plt.xlabel("Time (s)")
plt.ylabel("Steering Command")

plt.title(
    "Steering Command"
)

plt.grid()

plt.show()


# ============================================================
# DISTANCE TO TARGET
# ============================================================

plt.figure()

plt.plot(
    time_history,
    distance_history
)

plt.xlabel("Time (s)")
plt.ylabel("Distance to Target (m)")

plt.title(
    "Distance to Target"
)

plt.grid()

plt.show()