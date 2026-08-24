import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V7
# PREDICTED LANDING POINT GUIDANCE
# ============================================================


# ============================================================
# PARAMETERS
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
# WIND
# ============================================================

wind_x = 3.0
wind_y = 0.0


# ============================================================
# CONTROL PARAMETERS
# ============================================================

Kp = 0.8

max_turn_rate = np.radians(15.0)


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
print("VIRTUAL PARAFOIL V7")
print("----------------------------------------")

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
print("X:", target_x)
print("Y:", target_y)

print("----------------------------------------")

print("WIND")
print("X:", wind_x)
print("Y:", wind_y)

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

predicted_x_history = []
predicted_y_history = []

steering_history = []

prediction_error_history = []


# ============================================================
# MAIN LOOP
# ============================================================

while altitude > 0:

    # --------------------------------------------------------
    # REMAINING FLIGHT TIME
    # --------------------------------------------------------

    time_to_land = altitude / vertical_velocity


    # --------------------------------------------------------
    # CURRENT AIR VELOCITY
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
    # CURRENT GROUND VELOCITY
    # --------------------------------------------------------

    vx_ground = vx_air + wind_x
    vy_ground = vy_air + wind_y


    # --------------------------------------------------------
    # PREDICT LANDING POSITION
    # --------------------------------------------------------

    predicted_x = (
        x +
        vx_ground *
        time_to_land
    )

    predicted_y = (
        y +
        vy_ground *
        time_to_land
    )


    # --------------------------------------------------------
    # LANDING ERROR VECTOR
    # --------------------------------------------------------

    error_x = target_x - predicted_x
    error_y = target_y - predicted_y


    # --------------------------------------------------------
    # DESIRED CORRECTION DIRECTION
    # --------------------------------------------------------

    desired_heading = np.arctan2(
        error_y,
        error_x
    )


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
        Kp *
        heading_error
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
    # UPDATE POSITION
    # --------------------------------------------------------

    vx_air = (
        horizontal_air_velocity *
        np.cos(heading)
    )

    vy_air = (
        horizontal_air_velocity *
        np.sin(heading)
    )

    vx_ground = vx_air + wind_x
    vy_ground = vy_air + wind_y

    x += vx_ground * dt
    y += vy_ground * dt


    # --------------------------------------------------------
    # ALTITUDE
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

    predicted_x_history.append(
        predicted_x
    )

    predicted_y_history.append(
        predicted_y
    )

    steering_history.append(
        steering_command
    )

    prediction_error_history.append(
        np.sqrt(
            error_x**2 +
            error_y**2
        )
    )


    time += dt


# ============================================================
# FINAL ERROR
# ============================================================

landing_error = np.sqrt(
    (x - target_x)**2 +
    (y - target_y)**2
)


# ============================================================
# RESULTS
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
    label="Actual trajectory"
)

plt.plot(
    predicted_x_history,
    predicted_y_history,
    label="Predicted landing point"
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
    "V7 Predicted Landing Guidance"
)

plt.axis("equal")

plt.grid()

plt.legend()

plt.show()


# ============================================================
# PREDICTION ERROR
# ============================================================

plt.figure()

plt.plot(
    time_history,
    prediction_error_history
)

plt.xlabel("Time (s)")
plt.ylabel("Predicted Landing Error (m)")

plt.title(
    "Predicted Landing Error"
)

plt.grid()

plt.show()


# ============================================================
# STEERING COMMAND
# ============================================================

plt.figure()

plt.plot(
    time_history,
    steering_history
)

plt.xlabel("Time (s)")
plt.ylabel("Steering Command")

plt.title(
    "Guidance Steering Command"
)

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

plt.title(
    "Parafoil Heading"

)

plt.grid()

plt.show()