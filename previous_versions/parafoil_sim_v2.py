import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PARAFOIL PARAMETERS
# ============================================================

mass = 1.0              # kg
g = 9.81                # m/s^2

span = 1.6              # m
chord = 0.6             # m
area = span * chord      # m^2

rho = 1.225             # kg/m^3

# Initial aerodynamic coefficients
CL = 0.4
CD = 0.25


# ============================================================
# INITIAL CONDITIONS
# ============================================================

altitude = 600.0        # m

x = 0.0                 # m
y = 0.0                 # m


# ============================================================
# WIND
# ============================================================

wind_x = 0.0            # m/s
wind_y = 0.0            # m/s


# ============================================================
# AERODYNAMIC CALCULATIONS
# ============================================================

weight = mass * g


# Glide angle
glide_angle = np.arctan(CD / CL)


# Approximate steady-state airspeed
airspeed = np.sqrt(
    weight /
    (0.5 * rho * area * CL * np.cos(glide_angle))
)


# Horizontal and vertical velocity relative to air
horizontal_air_velocity = (
    airspeed * np.cos(glide_angle)
)

vertical_velocity = (
    airspeed * np.sin(glide_angle)
)


# ============================================================
# PRINT INITIAL RESULTS
# ============================================================

print("----------------------------------------")
print("VIRTUAL PARAFOIL V2")
print("----------------------------------------")

print("Parafoil area:", area, "m^2")
print("Weight:", weight, "N")
print("CL:", CL)
print("CD:", CD)

print("Glide angle:",
      np.degrees(glide_angle), "degrees")

print("Airspeed:",
      airspeed, "m/s")

print("Horizontal air velocity:",
      horizontal_air_velocity, "m/s")

print("Vertical descent velocity:",
      vertical_velocity, "m/s")

print("Glide ratio:",
      horizontal_air_velocity / vertical_velocity)

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


# ============================================================
# SIMULATION LOOP
# ============================================================

while altitude > 0:

    # --------------------------------------------------------
    # Ground velocity = air-relative velocity + wind
    # --------------------------------------------------------

    ground_velocity_x = horizontal_air_velocity + wind_x
    ground_velocity_y = wind_y


    # --------------------------------------------------------
    # Update position
    # --------------------------------------------------------

    x = x + ground_velocity_x * dt
    y = y + ground_velocity_y * dt

    altitude = altitude - vertical_velocity * dt


    # --------------------------------------------------------
    # Store data
    # --------------------------------------------------------

    time_history.append(time)
    x_history.append(x)
    y_history.append(y)
    altitude_history.append(altitude)


    time = time + dt


# ============================================================
# FINAL RESULTS
# ============================================================

print("Flight time:", time, "seconds")
print("Landing X:", x, "m")
print("Landing Y:", y, "m")

horizontal_distance = np.sqrt(x**2 + y**2)

print("Total horizontal distance:",
      horizontal_distance, "m")


# ============================================================
# ALTITUDE vs DISTANCE
# ============================================================

plt.figure()

plt.plot(x_history, altitude_history)

plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Altitude (m)")

plt.title("Parafoil Glide Trajectory")

plt.grid()

plt.show()


# ============================================================
# TOP VIEW
# ============================================================

plt.figure()

plt.plot(x_history, y_history)

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

plt.title("Parafoil Ground Track")

plt.axis("equal")

plt.grid()

plt.legend()

plt.show()