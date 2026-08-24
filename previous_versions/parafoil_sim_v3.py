import numpy as np
import matplotlib.pyplot as plt


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
# 0 degrees = positive X direction
heading = np.radians(0.0)


# ============================================================
# STEERING PARAMETERS
# ============================================================

# Steering command:
#
# -1.0 = maximum left
#  0.0 = no steering
# +1.0 = maximum right

steering_command = 0.30


# Maximum turning rate
max_turn_rate = np.radians(15.0)     # deg/s


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
    (0.5 * rho * area * CL * np.cos(glide_angle))
)

horizontal_air_velocity = (
    airspeed * np.cos(glide_angle)
)

vertical_velocity = (
    airspeed * np.sin(glide_angle)
)


# ============================================================
# TURN RATE
# ============================================================

turn_rate = max_turn_rate * steering_command


# ============================================================
# PRINT PARAMETERS
# ============================================================

print("----------------------------------------")
print("VIRTUAL PARAFOIL V3")
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
      horizontal_air_velocity / vertical_velocity)

print("Steering command:",
      steering_command)

print("Turn rate:",
      np.degrees(turn_rate), "deg/s")

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


# ============================================================
# SIMULATION LOOP
# ============================================================

while altitude > 0:

    # --------------------------------------------------------
    # Update heading
    # --------------------------------------------------------

    heading = heading + turn_rate * dt


    # --------------------------------------------------------
    # Calculate horizontal velocity
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
    # Add wind
    # --------------------------------------------------------

    ground_velocity_x = velocity_x + wind_x
    ground_velocity_y = velocity_y + wind_y


    # --------------------------------------------------------
    # Update position
    # --------------------------------------------------------

    x = x + ground_velocity_x * dt
    y = y + ground_velocity_y * dt


    # --------------------------------------------------------
    # Update altitude
    # --------------------------------------------------------

    altitude = altitude - vertical_velocity * dt


    # --------------------------------------------------------
    # Store history
    # --------------------------------------------------------

    time_history.append(time)

    x_history.append(x)
    y_history.append(y)

    altitude_history.append(altitude)

    heading_history.append(
        np.degrees(heading)
    )


    time = time + dt


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

print("Horizontal distance from deployment:",
      np.sqrt(x**2 + y**2), "m")

print("Final heading:",
      np.degrees(heading), "degrees")

print("----------------------------------------")


# ============================================================
# TRAJECTORY
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
    x_history[-1],
    y_history[-1],
    label="Landing"
)

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")

plt.title("V3 Parafoil Ground Track")

plt.axis("equal")

plt.grid()

plt.legend()

plt.show()


# ============================================================
# ALTITUDE PROFILE
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

plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Altitude (m)")

plt.title("Parafoil Descent Profile")

plt.grid()

plt.show()


# ============================================================
# HEADING
# ============================================================

plt.figure()

plt.plot(
    time_history,
    heading_history
)

plt.xlabel("Time (s)")
plt.ylabel("Heading (degrees)")

plt.title("Parafoil Heading")

plt.grid()

plt.show()