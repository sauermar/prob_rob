import numpy as np
import math

def gps_to_xy(lat, lon, lat0, lon0, radius):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    
    x = radius * (lon_rad - lon0_rad) * math.cos(lat0_rad)
    y = radius * (lat_rad - lat0_rad)
    
    return np.array([x, y])

def dms_to_decimal(degrees, minutes, seconds):
    return degrees + minutes / 60 + seconds / 3600

origin_lat = dms_to_decimal(50, 5, 18.404)
origin_lon = dms_to_decimal(14, 24, 15.307)

earth_r = 6378000 

# parameters from assignment
gps_accuracy = 8.0 
speed_accuracy = 0.6
process_noise_std = 0.05 
origin_accuracy = 0.5 

# Initial state vector [x, y, vx, vy]
x_ = np.array([0, 0, 0, 0])  # Robot starts at the origin with 0 velocity

# State transition matrix A 
A = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Control input matrix (no control input in this case)
B = np.array([[0], [0], [0], [0]])

# Measurement matrix C (we only measure position [x, y])
C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Initial covariance matrix E_ (uncertainty in position and velocity)
E_ = np.array([[origin_accuracy**2, 0, 0, 0],  # Uncertainty in x position
               [0, origin_accuracy**2, 0, 0],  # Uncertainty in y position
               [0, 0, speed_accuracy**2, 0],   # Uncertainty in x velocity
               [0, 0, 0, speed_accuracy**2]])  # Uncertainty in y velocity

# Process noise covariance matrix Q 
Q = np.array([[1/4*process_noise_std**2, 0, 1/2*process_noise_std**2, 0],
              [0, 1/4*process_noise_std**2, 0, 1/2*process_noise_std**2],
              [0, 0, process_noise_std**2, 0],
              [0, 0, 0, process_noise_std**2]])

# Measurement noise covariance matrix R (based on GPS accuracy)
R = np.eye(2) * gps_accuracy**2

estimated_positions = []

# Read GPS data from the file
with open('parking-data.txt', 'r') as file:
    gps_data = file.readlines()

# Kalman filter loop for each GPS measurement
for idx, line in enumerate(gps_data):
    lat, lon = map(float, line.strip().split(' '))
    z = gps_to_xy(lat, lon, origin_lat, origin_lon, earth_r)  # Measurement [x, y] from GPS

    # Prediction step
    x_ = A @ x_  # Predict the next state
    E_ = A @ E_ @ A.T + Q  # Predict the next error covariance

    # Measurement update (Kalman Gain)
    S = C @ E_ @ C.T + R  # Residual covariance
    K = E_ @ C.T @ np.linalg.inv(S)  # Kalman Gain

    # Update step
    y = z - C @ x_  # Innovation or residual
    x_ = x_ + K @ y  # Update state estimate
    I = np.eye(E_.shape[0])
    E_ = (I - K @ C) @ E_  # Update error covariance

    # Store every 10th position
    if idx % 10 == 0:
        estimated_positions.append(x_[:2])
        print(f"Estimated Position {idx}: x = {x_[0]:.2f} m, y = {x_[1]:.2f} m")

# Final estimated position
final_x, final_y = x_[:2]
print(f"\nFinal Estimated Position: x = {final_x} m, y = {final_y} m")

# Estimate uncertainty (standard deviation of position from covariance matrix)
position_uncertainty = np.sqrt(E_[0, 0] + E_[1, 1])
print(f"Final position uncertainty: ±{position_uncertainty:.2f} meters")

from scipy.stats import multivariate_normal

# Coordinates of the door in degrees
door_lat = np.radians(50 + 5/60 + 18.475/3600)
door_lon = np.radians(14 + 24/60 + 13.495/3600)

# Convert door's GPS coordinates to Cartesian (x, y) relative to the origin
door_position = gps_to_xy(door_lat, door_lon, origin_lat, origin_lon, earth_r)

final_position = np.array([final_x, final_y])
# Calculate distance between the final robot position and the door
distance_to_door = np.linalg.norm(final_position - door_position)
print(f"Distance to door: {distance_to_door} meters")

# Estimate probability using the 2D Gaussian distribution (mean = final_position, covariance = position_uncertainty)
rv = multivariate_normal(mean=final_position, cov=position_uncertainty)
# Calculate the probability density at the door's position
probability_at_door = rv.pdf(door_position)
print(f"Probability density at door position: {probability_at_door}")


