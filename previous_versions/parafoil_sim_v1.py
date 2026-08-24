import numpy as np
import matplotlib.pyplot as plt


# =========================
# PARAFOIL PARAMETERS
# =========================

mass = 1.0                 # kg
area = 0.96                # m^2
span = 1.6                 # m
chord = 0.6                # m

deployment_altitude = 600  # m

# Approximate velocities
vertical_speed = 4.0       # m/s downward
horizontal_speed = 6.0     # m/s


# =========================
# SIMULATION PARAMETERS
# =========================

dt = 0.1                   # time step
time = 0.0

x = 0.0                    # horizontal position
y = 0.0                    # horizontal position
altitude = deployment_altitude

x_history = []
y_history = []
altitude_history = []
time_history = []


# =========================
# SIMULATION LOOP
# =========================

while altitude > 0:

    # Horizontal motion
    x = x + horizontal_speed * dt

    # Vertical descent
    altitude = altitude - vertical_speed * dt

    # Store data
    x_history.append(x)
    y_history.append(y)
    altitude_history.append(altitude)
    time_history.append(time)

    time = time + dt


# =========================
# RESULTS
# =========================

print("Flight time:", time, "seconds")
print("Landing X:", x, "m")
print("Landing Y:", y, "m")


# =========================
# PLOT TRAJECTORY
# =========================

plt.figure()

plt.plot(x_history, altitude_history)

plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Altitude (m)")
plt.title("Virtual Parafoil Descent")

plt.grid()

plt.show()


# =========================
# TOP VIEW
# =========================

plt.figure()

plt.plot(x_history, y_history)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Parafoil Ground Track")

plt.axis("equal")
plt.grid()

plt.show()