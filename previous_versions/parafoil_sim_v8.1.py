import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VIRTUAL PARAFOIL V8
# TRAJECTORY-BASED GUIDANCE
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
# WIND
# ============================================================

wind_x = 3.0
wind_y = 0.0


# ============================================================
# CONTROL PARAMETERS
# ============================================================

max_turn_rate = np.radians(15.0)

# How often guidance chooses a new command
guidance_interval = 2.0


# ============================================================
# CANDIDATE STEERING COMMANDS
# ============================================================

candidate_commands = np.linspace(
    -1.0,
    1.0,
    21
)


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
# PRINT PARAMETERS
# ============================================================

print("----------------------------------------")
print("VIRTUAL PARAFOIL V8")
print("----------------------------------------")

print("Area:",
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

print("----------------------------------------")


# ============================================================
# FUNCTION: PREDICT LANDING
# ============================================================

def predict_landing(
    current_x,
    current_y,
    current_altitude,
    current_heading,
    steering_command
):

    # Remaining flight time

    remaining_time = (
        current_altitude /
        vertical_velocity
    )

    # Number of prediction steps

    prediction_dt = 0.5

    steps = int(
        remaining_time /
        prediction_dt
    )

    # Copy state

    px = current_x
    py = current_y

    pheading = current_heading

    # Simulate future trajectory

    for _ in range(steps):

        # Turn rate

        turn_rate = (
            max_turn_rate *
            steering_command
        )

        # Update heading

        pheading += (
            turn_rate *
            prediction_dt
        )

        # Keep heading in -pi to +pi

        pheading = (
            pheading + np.pi
        ) % (2 * np.pi) - np.pi

        # Air velocity

        vx_air = (
            horizontal_air_velocity *
            np.cos(pheading)
        )

        vy_air = (
            horizontal_air_velocity *
            np.sin(pheading)
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

        # Update position

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
# SIMULATION SETTINGS
# ============================================================

dt = 0.1

time = 0.0

next_guidance_update = 0.0


# ============================================================
# CURRENT STEERING
# ============================================================

current_steering = 0.0


# ============================================================
# HISTORY
# ============================================================

time_history = []

x_history = []
y_history = []

altitude_history = []

heading_history = []

steering_history = []

predicted_x_history = []
predicted_y_history = []

prediction_error_history = []

actual_distance_history = []


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

while altitude > 0:

    # ========================================================
    # GUIDANCE UPDATE
    # ========================================================

    if time >= next_guidance_update:

        best_command = 0.0

        best_error = float("inf")

        best_predicted_x = x

        best_predicted_y = y


        # ----------------------------------------------------
        # Test every candidate steering command
        # ----------------------------------------------------

        for command in candidate_commands:

            predicted_x, predicted_y = predict_landing(
                x,
                y,
                altitude,
                heading,
                command
            )


            # Landing error

            error = np.sqrt(
                (predicted_x - target_x)**2 +
                (predicted_y - target_y)**2
            )


            # Keep best command

            if error < best_error:

                best_error = error

                best_command = command

                best_predicted_x = predicted_x

                best_predicted_y = predicted_y


        # ----------------------------------------------------
        # Apply best command
        # ----------------------------------------------------

        current_steering = best_command

        predicted_x_current = best_predicted_x
        predicted_y_current = best_predicted_y

        predicted_error_current = best_error


        # Next guidance update

        next_guidance_update = (
            time +
            guidance_interval
        )


    # ========================================================
    # ACTUAL PARAFOIL DYNAMICS
    # ========================================================

    turn_rate = (
        max_turn_rate *
        current_steering
    )


    # Update heading

    heading += (
        turn_rate *
        dt
    )

    heading = (
        heading + np.pi
    ) % (2 * np.pi) - np.pi


    # --------------------------------------------------------
    # Air velocity
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
    # Ground velocity
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
    # Position update
    # --------------------------------------------------------

    x += (
        vx_ground *
        dt
    )

    y += (
        vy_ground *
        dt
    )


    # --------------------------------------------------------
    # Altitude update
    # --------------------------------------------------------

    altitude -= (
        vertical_velocity *
        dt
    )


    # ========================================================
    # STORE DATA
    # ========================================================

    time_history.append(time)

    x_history.append(x)

    y_history.append(y)

    altitude_history.append(altitude)

    heading_history.append(
        np.degrees(heading)
    )

    steering_history.append(
        current_steering
    )

    predicted_x_history.append(
        predicted_x_current
    )

    predicted_y_history.append(
        predicted_y_current
    )

    prediction_error_history.append(
        predicted_error_current
    )

    actual_distance_history.append(
        np.sqrt(
            (x - target_x)**2 +
            (y - target_y)**2
        )
    )


    # ========================================================
    # TIME
    # ========================================================

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

print("Final steering:",
      current_steering)

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
    label="Predicted landing"
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
    "V8 Trajectory-Based Parafoil Guidance"
)

plt.axis("equal")

plt.grid()

plt.legend()

plt.show()


# ============================================================
# ACTUAL DISTANCE TO TARGET
# ============================================================

plt.figure()

plt.plot(
    time_history,
    actual_distance_history
)

plt.xlabel("Time (s)")
plt.ylabel("Actual Distance to Target (m)")

plt.title(
    "Actual Distance to Target"
)

plt.grid()

plt.show()


# ============================================================
# PREDICTED LANDING ERROR
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

plt.step(
    time_history,
    steering_history,
    where="post"
)

plt.xlabel("Time (s)")
plt.ylabel("Steering Command")

plt.title(
    "Optimal Steering Command"
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

# ============================================================
# V8.1 CONTROLLER ANALYSIS
# ============================================================

steering_array = np.array(steering_history)

max_steering = np.max(
    np.abs(steering_array)
)

average_steering = np.mean(
    np.abs(steering_array)
)

steering_reversals = 0

for i in range(1, len(steering_array)):

    if (
        steering_array[i] != 0
        and steering_array[i - 1] != 0
        and
        np.sign(steering_array[i])
        !=
        np.sign(steering_array[i - 1])
    ):
        steering_reversals += 1

print()
print("----------------------------------------")
print("V8.1 CONTROLLER ANALYSIS")
print("----------------------------------------")

print("Maximum steering magnitude:", max_steering)
print("Average absolute steering:", average_steering)
print("Steering reversals:", steering_reversals)

print("----------------------------------------")