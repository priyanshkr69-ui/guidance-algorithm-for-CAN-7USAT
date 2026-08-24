import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V4
# TARGET-BASED GUIDANCE
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

# Initial heading
# 0 degrees = +X direction

heading = np.radians(0.0)


# ============================================================
# TARGET LOCATION
# ============================================================

target_x = 500.0
target_y = 200.0


# ============================================================
# CONTROLLER PARAMETERS
# ============================================================

# Maximum turn rate

max_turn_rate = np.radians(15.0)


# Proportional controller gain

Kp = 1.5


# ============================================================
# WIND
# ============================================================

wind_x = 0.0
wind_y = 0.0


# ============================================================
# AERODYNAMIC CALCULATIONS
# ============================================================

weight = mass * g

glide_angle = np.arctan(CD / CL)


airspeed = np.sqrt(
    weight /
    (
        0.5
        * rho
        * area
        * CL
        * np.cos(glide_angle)
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
# PRINT INITIAL PARAMETERS
# ============================================================

print("----------------------------------------")
print("VIRTUAL PARAFOIL V4")
print("----------------------------------------")

print("Parafoil area:",
      area, "m^2")

print("Mass:",
      mass, "kg")

print("CL:",
      CL)

print("CD:",
      CD)

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

print("Target X:",
      target_x, "m")

print("Target Y:",
      target_y, "m")

print("----------------------------------------")


# ============================================================
# SIMULATION SETTINGS
# ============================================================

dt = 0.1

time = 0.0


# ============================================================
# HISTORY ARRAYS
# ============================================================

time_history = []

x_history = []
y_history = []

altitude_history = []

heading_history = []

steering_history = []

heading_error_history = []

distance_to_target_history = []


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

while altitude > 0:

    # --------------------------------------------------------
    # 1. VECTOR FROM PARAFOIL TO TARGET
    # --------------------------------------------------------

    dx = target_x - x
    dy = target_y - y


    # --------------------------------------------------------
    # 2. DISTANCE TO TARGET
    # --------------------------------------------------------

    distance_to_target = np.sqrt(
        dx**2 + dy**2
    )


    # --------------------------------------------------------
    # 3. DESIRED HEADING
    # --------------------------------------------------------

    desired_heading = np.arctan2(
        dy,
        dx
    )


    # --------------------------------------------------------
    # 4. HEADING ERROR
    # --------------------------------------------------------

    heading_error = (
        desired_heading - heading
    )


    # --------------------------------------------------------
    # WRAP ANGLE TO -PI ... +PI
    # --------------------------------------------------------

    heading_error = (
        heading_error + np.pi
    ) % (2 * np.pi) - np.pi


    # --------------------------------------------------------
    # 5. PROPORTIONAL GUIDANCE CONTROLLER
    # --------------------------------------------------------

    steering_command = Kp * heading_error


    # --------------------------------------------------------
    # LIMIT STEERING COMMAND
    # --------------------------------------------------------

    steering_command = np.clip(
        steering_command,
        -1.0,
        1.0
    )


    # --------------------------------------------------------
    # 6. CONVERT STEERING TO TURN RATE
    # --------------------------------------------------------

    turn_rate = (
        max_turn_rate *
        steering_command
    )


    # --------------------------------------------------------
    # 7. UPDATE HEADING
    # --------------------------------------------------------

    heading = (
        heading +
        turn_rate * dt
    )


    # --------------------------------------------------------
    # 8. HORIZONTAL VELOCITY
    # --------------------------------------------------------

    velocity_x = (
        horizontal_air_velocity *
        np.cos(heading)
    )

    velocity_y = (
        horizontal_air_velocity *
        np.sin(heading)
    )


    # --------------------------------------------------------
    # 9. ADD WIND
    # --------------------------------------------------------

    ground_velocity_x = (
        velocity_x +
        wind_x
    )

    ground_velocity_y = (
        velocity_y +
        wind_y
    )


    # --------------------------------------------------------
    # 10. UPDATE POSITION
    # --------------------------------------------------------

    x = (
        x +
        ground_velocity_x * dt
    )

    y = (
        y +
        ground_velocity_y * dt
    )


    # --------------------------------------------------------
    # 11. UPDATE ALTITUDE
    # --------------------------------------------------------

    altitude = (
        altitude -
        vertical_velocity * dt
    )


    # --------------------------------------------------------
    # 12. STORE DATA
    # --------------------------------------------------------

    time_history.append(time)

    x_history.append(x)

    y_history.append(y)

    altitude_history.append(altitude)

    heading_history.append(
        np.degrees(heading)
    )

    steering_history.append(
        steering_command
    )

    heading_error_history.append(
        np.degrees(heading_error)
    )

    distance_to_target_history.append(
        distance_to_target
    )


    # --------------------------------------------------------
    # UPDATE TIME
    # --------------------------------------------------------

    time = time + dt


# ============================================================
# FINAL LANDING ERROR
# ============================================================

landing_error = np.sqrt(
    (x - target_x)**2 +
    (y - target_y)**2
)


# ============================================================
# FINAL RESULTS
# ============================================================

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
# PLOT 1 — GROUND TRACK
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
    "V4 Parafoil Ground Track"
)

plt.axis("equal")

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PLOT 2 — ALTITUDE
# ============================================================

plt.figure()

horizontal_distance_history = np.sqrt(
    np.array(x_history)**2 +
    np.array(y_history)**2
)

plt.plot(
    horizontal_distance_history,
    altitude_history
)

plt.xlabel(
    "Horizontal Distance (m)"
)

plt.ylabel(
    "Altitude (m)"
)

plt.title(
    "Parafoil Descent Profile"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 3 — HEADING
# ============================================================

plt.figure()

plt.plot(
    time_history,
    heading_history
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
# PLOT 4 — STEERING COMMAND
# ============================================================

plt.figure()

plt.plot(
    time_history,
    steering_history
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Steering Command"
)

plt.title(
    "Guidance Steering Command"
)

plt.grid()

plt.show()


# ============================================================
# PLOT 5 — DISTANCE TO TARGET
# ============================================================

plt.figure()

plt.plot(
    time_history,
    distance_to_target_history
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

plt.show()