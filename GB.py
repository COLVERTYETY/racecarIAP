import sys
sys.path.insert(0, '../library')
import racecar_core
import json
import _pickle as cPickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd

GB_model = None
scaler = None

# Initialize RACECAR
rc = racecar_core.create_racecar()

# Driving parameters
speed = 0.0
angle = 0.0



# Start function
def start():
    rc.drive.stop()
    

# Update function
def update():
    lidar_samples = rc.lidar.get_samples().tolist()
    # imu_angle = rc.physics.get_angular_velocity().tolist()
    # imu_acc = rc.physics.get_linear_acceleration().tolist()

    lidar_features = pd.DataFrame([lidar_samples])
    # imu_acc_features = pd.DataFrame([imu_acc])
    # imu_angle_features = pd.DataFrame([imu_angle])

    # print(f"lidar_features: {lidar_features.shape}, imu_acc_features: {imu_acc_features.shape}, imu_angle_features: {imu_angle_features.shape}")

    # features = pd.concat([lidar_features, imu_acc_features, imu_angle_features], axis=1)
    # print(f"features: {features.shape}")
    features = lidar_features
    features = scaler.transform(features)
    prediction = GB_model.predict(features)
    angle = prediction[0]
    angle = max(min(angle, 1.0), -1.0)
    print("Angle: ", angle)
    rc.drive.set_speed_angle(1.0, angle)



def update_slow():
    # Debug info
    print("Speed: ", speed)
    print("Angle: ", angle)

# Main execution
if __name__ == "__main__":
    with open('scaler.pkl', 'rb') as f:
        scaler = cPickle.load(f)
    with open('GB.pkl', 'rb') as f:
        GB_model = cPickle.load(f)
        GB_model.verbose = 0
    rc.set_start_update(start, update, update_slow)
    rc.go()
