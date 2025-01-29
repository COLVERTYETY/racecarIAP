import sys
sys.path.insert(0, '../library')
import racecar_core
import _pickle as cPickle
import os
import numpy as np
import time
import json
# from sklearn.preprocessing import StandardScaler

# If you installed pycoral:
# from pycoral.utils.edgetpu import make_interpreter
# from pycoral.adapters import common
import tflite_runtime.interpreter as tflite

# -- Global objects --
LSTM_model = None
scaler = None

# Adjust to match your LSTM sequence length
SEQ_LEN = 60  

# Rolling queue for storing the latest LiDAR frames
frameQ = []
# For optional debugging or smoothing
angles = []
fixspeed = 0.6

# Initialize RACECAR
rc = racecar_core.create_racecar()
lastts = time.time()
fps=0
# Driving parameters
speed = 0.0
angle = 0.0


class Scaler():
    def __init__(self, path=None):
        self.path = path
        if path is not None:
            with open(path, 'r') as f:
                data = json.load(f)
                self.mean = np.array(data["mean"]).astype(np.float32)
                self.std = np.array(data["std"]).astype(np.float32)
                print("loaded sclaer of shape:")
                print("mean:", self.mean.shape, self.mean.dtype)
                print("std :", self.std.shape, self.mean.dtype)
        else:
            self.mean = None
            self.std = None

    def transform(self, data):
        return (data - self.mean) / self.std

def remove_center_cone(lidar_data, cone_angle=45):
    num_samples = len(lidar_data)
    center_index = num_samples // 2
    cone_samples = int((cone_angle / 360) * num_samples)
    t= np.concatenate((lidar_data[:center_index - cone_samples // 2], lidar_data[center_index + cone_samples // 2:]))
    return t

def start():
    """
    Called once at the start: good place to initialize or reset.
    """
    rc.drive.stop()

def update():
    """
    Called every frame at the real-time rate (~60 Hz).
    """
    global speed, angle, frameQ, LSTM_model, scaler, angles, lastts, fps

    # 1) Acquire LiDAR data
    lidar_samples = rc.lidar.get_samples().tolist()

    # 2) Clean/replace inf or NaN
    lidar_features = np.array(lidar_samples)
    lidar_features[np.isinf(lidar_features)] = np.nan
    lidar_features = np.nan_to_num(lidar_features)
    lidar_features = remove_center_cone(lidar_features)
    # 3) Each frame is shape (1, 946) if your model expects 946 features per frame
    #    Adjust if you have a different dimension per frame
    lidar_features = lidar_features.reshape(1, -1).astype(np.float32)
    
    # 4) Append this frame to our rolling queue
    frameQ.append( lidar_features)
    if len(frameQ) > SEQ_LEN:
        frameQ.pop(0)

    # 5) Only run inference if we have at least SEQ_LEN frames
    if len(frameQ) == SEQ_LEN:
        # Combine frames into shape (SEQ_LEN, 946)
        seq_data = np.concatenate(frameQ, axis=0)  # (60, 946)
        # print(seq_data.shape)
        # Apply the same scaler used during training
        seq_data = scaler.transform(seq_data).astype(np.float32)

        # Add batch dimension -> (1, SEQ_LEN, 946)
        seq_data = np.expand_dims(seq_data, axis=0)  # (1, 60, 946)

        # 6) Run inference on the Edge TPU
        LSTM_model.set_tensor(input_details[0]['index'], seq_data)
        LSTM_model.invoke()
        prediction = LSTM_model.get_tensor(output_details[0]['index'])  # shape (1, 1) or (1,) depending on your model
        print(f"Prediction  {prediction}")
        # Extract predicted steering angle
        angle_pred = float(prediction[0, 0] if prediction.ndim == 2 else prediction[0])
        # print(f"Predicted angle: {angle_pred:.3f}")
        # Clamp angle to [-1, 1]
        angle = max(min(angle_pred, 1.0), -1.0)

        # Optional small moving average or debugging
        angles.append(angle)
        if len(angles) > SEQ_LEN:
            angles.pop(0)
        # Example: Check MSE across last 10 angles vs. the current prediction
        # (purely optional debugging snippet)
        # if len(angles) == 10:
        #     mse = np.mean((np.array(angles) - angle_pred)**2)
        #     if mse > 1:
        #         print("Warning: High MSE on last predictions:", mse)

    else:
        # Not enough frames yet
        angle = 0.0
        print(f"Waiting for frames: {len(frameQ)}/{SEQ_LEN}")

    # 7) Collision detection or other logic
    imu_acc = rc.physics.get_linear_acceleration()
    if imu_acc[2] < -1:
        print("Collision detected!", imu_acc[2])

    # 8) Set speed and angle
    rc.drive.set_speed_angle(fixspeed, angle)
    current = time.time()
    fps = 1/(current - lastts)
    lastts = current


def update_slow():
    global angle, fps
    """
    Called at a slower rate. Good place for printing debug info.
    """
    print(f"Steering Angle: {angle:.3f} FPS: {fps:.2f}")


# Main execution
if __name__ == "__main__":
    # 1) Load the scaler (must match the one used during training)
    scaler = Scaler("LSTM_scaler.json")

    # 2) Create interpreter for your LSTM TFLite model
    tflite_path = "LSTM_steering.tflite"  # adapt to your actual filename
    LSTM_model = tflite.Interpreter(
    model_path="LSTM_steering.tflite",
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    LSTM_model.allocate_tensors()

    # 3) Get input/output details to set/get data later
    input_details = LSTM_model.get_input_details()
    output_details = LSTM_model.get_output_details()
    print("Model Input Shape:", input_details[0]['shape'])
    print("Model Output Shape:", output_details[0]['shape'])

    # 4) Launch the RACECAR loop
    rc.set_start_update(start, update, update_slow)
    rc.go()
