import sys
sys.path.insert(0, '../library')
import racecar_core
import pandas as pd
import json
import os


# Initialize RACECAR
rc = racecar_core.create_racecar()
LOGFILE = "test2.csv"
df = pd.DataFrame(columns=["ts", "lidar", "imu_acc", "imu_angle", "speed", "angle"])
tcounter = 0
itcounter = 0

# Driving parameters
speed = 0.0
angle = 0.0
speed_step = 0.1
max_speed = 1.0
angle_step = 0.1
max_angle = 1.0

# Function to handle user input
def handle_input():
    global speed, angle
    
    # Forward and backward movement
    if rc.controller.get_trigger(rc.controller.Trigger.RIGHT) > 0.1:
        speed = min(speed + speed_step, max_speed)
    elif rc.controller.get_trigger(rc.controller.Trigger.LEFT) > 0.1:
        speed = max(speed - speed_step, -max_speed)
    else:
        speed *= 0.9  # Gradual slowdown when no input is given
    
    # Steering
    left_joystick = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    if left_joystick[0] > 0.1:
        angle = min(angle + angle_step, max_angle)
    elif left_joystick[0] < -0.1:
        angle = max(angle - angle_step, -max_angle)
    else:
        angle *= 0.9  # Gradual centering of the steering
    
    # Braking
    if rc.controller.was_pressed(rc.controller.Button.B):
        speed = 0.0

# Start function
def start():
    rc.drive.stop()
    

# Update function
def update():
    global tcounter, df
    handle_input()
    rc.drive.set_speed_angle(speed, angle)
    samples = rc.lidar.get_samples().tolist()
    imu_angle = rc.physics.get_angular_velocity().tolist()
    imu_acc = rc.physics.get_linear_acceleration().tolist()
    tcounter += rc.get_delta_time()
    
    new_data = {
        "ts": tcounter,
        "lidar": json.dumps(samples),
        "imu_acc": json.dumps(imu_acc),
        "imu_angle": json.dumps(imu_angle),
        "speed": speed,
        "angle": angle
    }
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)


def update_slow():
    global df, itcounter
    # df.to_csv(LOGFILE, index=False)
    itcounter += 1
    if itcounter % 10 == 0:
        df.to_csv(LOGFILE, mode='a', header=not os.path.exists(LOGFILE), index=False)
        df = df.iloc[0:0]  # Clear the DataFrame
        print("Appended to CSV and cleared DataFrame")
    
    # Debug info
    print("Speed: ", speed)
    print("Angle: ", angle)
    print("Left Trigger: ", rc.controller.get_trigger(rc.controller.Trigger.LEFT))
    print("Right Trigger: ", rc.controller.get_trigger(rc.controller.Trigger.RIGHT))
    print("Left Joystick: ", rc.controller.get_joystick(rc.controller.Joystick.LEFT))
    print("Right Joystick: ", rc.controller.get_joystick(rc.controller.Joystick.RIGHT))
    print("X Button: ", rc.controller.was_pressed(rc.controller.Button.X))
    print("Y Button: ", rc.controller.was_pressed(rc.controller.Button.Y))
    print("A Button: ", rc.controller.was_pressed(rc.controller.Button.A))
    print("B Button: ", rc.controller.was_pressed(rc.controller.Button.B))

# Main execution
if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
