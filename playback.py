#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

def main():
    # Load CSV
    df = pd.read_csv("test2.csv")
    
    # Parse JSON columns
    df["lidar"] = df["lidar"].apply(json.loads)
    df["imu_acc"] = df["imu_acc"].apply(json.loads)
    df["imu_angle"] = df["imu_angle"].apply(json.loads)
    
    # ----------------------------------
    # Optionally, still plot Speed/Angle
    # ----------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(df["ts"], df["speed"], label="Speed")
    plt.plot(df["ts"], df["angle"], label="Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Speed and Steering Angle vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("speed_angle_plot.png")
    plt.close()
    
    # ----------------------------------
    # Optionally, still plot IMU data
    # ----------------------------------
    # IMU Acceleration
    imu_acc_matrix = np.array(df["imu_acc"].tolist())  # shape: (n_frames, 3)
    plt.figure(figsize=(10, 5))
    plt.plot(df["ts"], imu_acc_matrix[:, 0], label="Acc X")
    plt.plot(df["ts"], imu_acc_matrix[:, 1], label="Acc Y")
    plt.plot(df["ts"], imu_acc_matrix[:, 2], label="Acc Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("IMU Acceleration vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("imu_acc_plot.png")
    plt.close()

    # IMU Angular Velocity
    imu_ang_matrix = np.array(df["imu_angle"].tolist())  # shape: (n_frames, 3)
    plt.figure(figsize=(10, 5))
    plt.plot(df["ts"], imu_ang_matrix[:, 0], label="Ang Vel X")
    plt.plot(df["ts"], imu_ang_matrix[:, 1], label="Ang Vel Y")
    plt.plot(df["ts"], imu_ang_matrix[:, 2], label="Ang Vel Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.title("IMU Angular Velocity vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("imu_ang_plot.png")
    plt.close()
    
    # ---------------------------------------------------
    #  Visualize ALL LIDAR frames as a 2D "image"
    # ---------------------------------------------------
    # Convert list of lists into a 2D array:
    # shape will be (#_of_frames, #_of_lidar_samples)
    lidar_matrix = np.array(df["lidar"].tolist())
    
    # Plot as an image with each row = 1 frame, each column = 1 sample
    plt.figure(figsize=(10, 6))
    plt.imshow(lidar_matrix, cmap="jet", aspect="auto")
    plt.colorbar(label="Distance")
    plt.xlabel("Sample Index")
    plt.ylabel("Frame Index")
    plt.title("LIDAR Data (All Frames)")
    plt.savefig("lidar_data_image.png")
    plt.close()

if __name__ == "__main__":
    main()
