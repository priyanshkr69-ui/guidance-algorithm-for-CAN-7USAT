import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V6
# WIND-COMPENSATED GUIDANCE
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
print("VIRTUAL PARAFOIL V6")
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

wind_speed = np.sqrt(
    wind_x**2 + wind_y**2
)

print("Wind speed:",
      wind_speed, "m/s")

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

steering_history = []

heading_error_history = []

distance_to_target_history = []

ground_speed_history = []


# ============================================================
# MAIN LOOP
# ============================================================

while altitude > 0:

    # --------------------------------------------------------
    # 1. VECTOR FROM PARAFOIL TO TARGET
    # --------------------------------------------------------

    dx = target_x - x
    dy = target_y - y

    distance_to_target = np.sqrt(
        dx**2 + dy**2
    )


    # --------------------------------------------------------
    # 2. TARGET BEARING
    # --------------------------------------------------------

    target_bearing = np.arctan2(
        dy,
        dx
    )


    # --------------------------------------------------------
    # 3. WIND COMPONENT PERPENDICULAR TO TARGET
    # --------------------------------------------------------

    wind_perpendicular = (
        -wind_x * np.sin(target_bearing)
        + wind_y * np.cos(target_bearing)
    )


    # --------------------------------------------------------
    # 4. CALCULATE WIND CORRECTION
    # --------------------------------------------------------

    ratio = (
        -wind_perpendicular /
        horizontal_air_velocity
    )


    # Prevent invalid arcsin

    ratio = np.clip(
        ratio,
        -1.0,
        1.0
    )


    crab_angle = np.arcsin(ratio)


    # --------------------------------------------------------
    # 5. WIND-COMPENSATED DESIRED HEADING
    # --------------------------------------------------------

    desired_heading = (
        target_bearing +
        crab_angle
    )


    # --------------------------------------------------------
    # 6. HEADING ERROR
    # --------------------------------------------------------

    heading_error = (
        desired_heading -
        heading
    )


    # Wrap to -pi ... +pi

    heading_error = (
        heading_error + np.pi
    ) % (2 * np.pi) - np.pi


    # --------------------------------------------------------
    # 7. PROPORTIONAL CONTROLLER
    # --------------------------------------------------------

    steering_command = (
        Kp * heading_error
    )


    # Limit steering

    steering_command = np.clip(
        steering_command,
        -1.0,
        1.0
    )


    # --------------------------------------------------------
    # 8. TURN RATE
    # --------------------------------------------------------

    turn_rate = (
        max_turn_rate *
        steering_command
    )


    # --------------------------------------------------------
    # 9. UPDATE HEADING
    # --------------------------------------------------------

    heading = (
        heading +
        turn_rate * dt
    )


    # Wrap heading

    heading = (
        heading + np.pi
    ) % (2 * np.pi) - np.pi


    # --------------------------------------------------------
    # 10. AIR VELOCITY
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
    # 11. GROUND VELOCITY
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
    # 12. GROUND SPEED
    # --------------------------------------------------------

    ground_speed = np.sqrt(
        ground_velocity_x**2 +
        ground_velocity_y**2
    )


    # --------------------------------------------------------
    # 13. UPDATE POSITION
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
    # 14. UPDATE ALTITUDE
    # --------------------------------------------------------

    altitude = (
        altitude -
        vertical_velocity * dt
    )


    # --------------------------------------------------------
    # 15. STORE HISTORY
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
    "V6 Wind-Compensated Parafoil Ground Track"
)

plt.axis("equal")

plt.grid()

plt.legend()

plt.show()


# ============================================================
# ALTITUDE
# ============================================================

plt.figure()

plt.plot(
    np.sqrt(
        np.array(x_history)**2 +
        np.array(y_history)**2
    ),
    altitude_history
)

plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Altitude (m)")

plt.title("Parafoil Descent Profile")

plt.grid()

plt.show()


# ============================================================
# HEADING COMPARISON
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

plt.xlabel("Time (s)")
plt.ylabel("Heading (degrees)")

plt.title("Actual vs Desired Heading")

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

plt.title("Wind-Compensated Steering Command")

plt.grid()

plt.show()


# ============================================================
# DISTANCE TO TARGET
# ============================================================

plt.figure()

plt.plot(
    time_history,
    distance_to_target_history
)

plt.xlabel("Time (s)")
plt.ylabel("Distance to Target (m)")

plt.title("Distance to Target")

plt.grid()

plt.show()