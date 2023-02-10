
import matplotlib.pyplot as plt
import numpy as np

trip_distance = 6.0 # km
trip_duration = 12.0 # minutes
trip_avg_speed = 30.0 # km/h

# trip duration in minutes
def duration(distance, speed):
    return distance * 1/speed * 60.0

def speed_error(speed):
    dur = duration(trip_distance, speed)
    return trip_duration - dur

# Approximate diff using limit definition
def speed_error_num_diff(speed, delta=1e-3):
    return (speed_error(speed+delta)**2 - speed_error(speed)**2) / delta

# Compute diff using exact formula
def speed_error_diff(speed, delta=1e-3):
    return 2 * trip_distance*60.0/speed**2 * speed_error(speed)

speeds = np.linspace(20, 50)
error = np.vectorize(speed_error)(speeds)
squared_error = error**2
tangente = speed_error(35)**2 + speed_error_diff(35) * (np.linspace(20, 50) - 35)

fig, ax = plt.subplots()
ax.grid(True, which='both')

ax.plot(speeds, squared_error, label='Squared error wrt to speed param')
ax.scatter([trip_avg_speed, 35], [0, speed_error(35)**2])
ax.plot(np.linspace(20, 50), tangente, label='Tangent')

plt.xlabel('average speed')
plt.legend()
plt.show()