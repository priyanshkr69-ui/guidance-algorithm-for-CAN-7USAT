import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V5
# TARGET GUIDANCE + WIND
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
# TARGET
# ============================================================

target_x = 500.0
target_y = 200.0


# ============================================================
# GUIDANCE CONTROLLER
# ============================================================

Kp = 1.5

max_turn_rate = np.radians(15.0)


# ============================================================
# WIND
# ============================================================

# Wind velocity in m/s
#
# Positive X = wind towards +X
# Positive Y = wind towards +Y

wind_x = 3.0
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
# PRINT PARAMETERS
# ============================================================

print("----------------------------------------")
print("VIRTUAL PARAFOIL V5")
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

print("WIND")

print("Wind X:",
      wind_x, "m/s")

print("Wind Y:",
      wind_y, "m/s")

print("Wind speed:",
      np.sqrt(wind_x**2 + wind_y**2),
      "m/s")

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

steering_history = []

heading_error_history = []

distance_to_target_history = []

ground_speed_history = []


# ============================================================
# MAIN LOOP
# ============================================================

while altitude > 0:

    # --------------------------------------------------------
    # 1. POSITION ERROR TO TARGET
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


    # Wrap angle between -pi and +pi

    heading_error = (
        heading_error + np.pi
    ) % (2 * np.pi) - np.pi


    # --------------------------------------------------------
    # 5. PROPORTIONAL CONTROLLER
    # --------------------------------------------------------

    steering_command = Kp * heading_error


    # Limit steering

    steering_command = np.clip(
        steering_command,
        -1.0,
        1.0
    )


    # --------------------------------------------------------
    # 6. TURN RATE
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

    # Keep heading between -pi and +pi

    heading = (
        heading + np.pi
    ) % (2 * np.pi) - np.pi


    # --------------------------------------------------------
    # 8. AIR-RELATIVE VELOCITY
    # --------------------------------------------------------

    velocity_air_x = (
        horizontal_air_velocity *
        np.cos(heading)
    )

    velocity_air_y = (
        horizontal_air_velocity *
        np.sin(heading)
    )


    # --------------------------------------------------------
    # 9. ADD WIND
    # --------------------------------------------------------

    ground_velocity_x = (
        velocity_air_x +
        wind_x
    )

    ground_velocity_y = (
        velocity_air_y +
        wind_y
    )


    # --------------------------------------------------------
    # 10. GROUND SPEED
    # --------------------------------------------------------

    ground_speed = np.sqrt(
        ground_velocity_x**2 +
        ground_velocity_y**2
    )


    # --------------------------------------------------------
    # 11. UPDATE POSITION
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
    # 12. UPDATE ALTITUDE
    # --------------------------------------------------------

    altitude = (
        altitude -
        vertical_velocity * dt
    )


    # --------------------------------------------------------
    # 13. SAVE HISTORY
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

    ground_speed_history.append(
        ground_speed
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time += dt


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
    "V5 Parafoil Ground Track with Wind"
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


# ============================================================
# PLOT 6 — GROUND SPEED
# ============================================================

plt.figure()

plt.plot(
    time_history,
    ground_speed_history
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Ground Speed (m/s)"
)

plt.title(
    "Parafoil Ground Speed"

)

plt.grid()

plt.show()
