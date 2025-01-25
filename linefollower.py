import sys
sys.path.insert(0, '../library')
import racecar_core
import numpy as np
import json
import os
import numpy as np
import cv2 as cv
from racecar_utils import get_lidar_average_distance

# Tunable parameters for "look-ahead" and speed control:
LOOKAHEAD_FRACTION = 0.5     # Use top 50% of the cropped image for the centroid
BASE_SPEED         = 0.6     # Base speed (when error = 0)
SPEED_K            = 0.002   # Scaling factor for proportional speed control
MIN_SPEED          = 0.2     # Minimum speed clamp
MAX_SPEED          = 1.0     # Maximum speed clamp


class PID:
    def __init__(self, Kp, Ki, Kd, buffer_length):
        self.kp = Kp
        self.kd = Kd
        self.ki = Ki
        self.buffer_length = buffer_length
        self.error_buffer = [0] * buffer_length

    def forward(self, error, verbose=False):
        # Update the error buffer
        self.error_buffer.pop(0)
        self.error_buffer.append(error)
        
        # Calculate the proportional term
        P = self.kp * error

        # Calculate the integral term using the error buffer
        I = self.ki * sum(self.error_buffer)

        # Calculate the derivative term using numpy
        # Fit a linear polynomial (degree=1) to the error buffer
        indices = np.arange(len(self.error_buffer))
        coefficients = np.polyfit(indices, self.error_buffer, 1)
        
        # The derivative (slope) is the first coefficient in the linear fit
        derivative = coefficients[0]
        D = self.kd * derivative

        if verbose:
            print(f"P: {P}, I: {I}, D: {D}, Output: {P + I + D}")
        
        # Return the combined PID output
        return P + I - D


#############" GLOBAL VARIABLES"###############

# Initialize RACECAR
rc = racecar_core.create_racecar()
speed = 0
angle = 0
dist_lp = 0
cX_lp = 0

# angle_PID = PID(0.00025, 0.0001, 0.00005, 10)
angle_PID = PID(0.00032, 0.0001, 0.00005, 10)

img_counter = 0

hsv_low = np.array([101, 36, 119])
hsv_high = np.array([131, 255, 255])

crop_top_left = (0, 240) #240
# crop_bottom_right = (640, 480)
crop_bottom_right = (640, 400)

  
# Store last time distance

# Start function
def start():
    rc.drive.set_speed_angle(0, 0)
    

# Update function
def update():
    global dist_lp, speed, angle, angle_PID, img_counter, cX_lp

    samples = rc.lidar.get_samples()
    dist = get_lidar_average_distance(samples, 0, 5)
    dist_lp = dist_lp*0.15 + dist*0.85 


    color_image = rc.camera.get_color_image()

    if color_image is not None:
        
        # Crop the image
        cropped_image = color_image[crop_top_left[1]:crop_bottom_right[1], crop_top_left[0]:crop_bottom_right[0]]
        hsv_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_image, hsv_low, hsv_high)
        #  find cluster pixels that are connected and closest to teh center of teh crop rectangle
        M = cv.moments(mask)
        if M["m00"] > 1000:
            cX = int(M["m10"] / M["m00"])
            cX_lp = cX_lp*0.05 + cX*0.95
            # print("cX", cX)
            cY = int(M["m01"] / M["m00"])
            # Draw the center of the contour on the image
            cv.circle(mask, (cX, cY), 7, (255, 255, 255), -1)
            # Calculate error from center
            error = cX_lp - mask.shape[1] // 2
            error *= 1+0.5*(mask.shape[0]-cY)/mask.shape[0]
            # speed = 0.5
            # speed = 0.65
            speed = 0.65
            # if cY<mask.shape[0]//2:
            #     speed = 0.4
            # speed = 0.85*mask.shape[0]/cY
            
            errorN = error / (mask.shape[1] // 2)
            # speed = 0.75*(1-abs(errorN)) + 0.5
            # if sum(mask[0])>10:
            # Apply PID controller
            angle = angle_PID.forward(error, verbose = True)
            # print("angle", angle)
            angle = max(min(angle, 1.0),-1.0)
            speed = max(min(speed, 1.0),-1.0)
            print(speed)
            # else:
                # angle = 1
            # print("after clamp", angle)
            if rc.controller.is_down(rc.controller.Button.X):
                # print(img_counter)
                img_counter+=1
                filename = f"{img_counter}_img.jpg"
                cv.imwrite(filename, cropped_image)
                print(f"Saved {filename}")
                filename = f"{img_counter}_mask.jpg"
                cv.imwrite(filename, mask)
        else:
            angle +=0.3
            angle = max(min(angle, 1.0),-1.0)
        rc.drive.set_speed_angle(speed, angle)

    
def update_slow():
    # slow update at ~10hz
    global img_counter, angle, speed
    # print(f"saved a total of {img_counter} images")
    print(angle)

    



# Main execution
if __name__ == "__main__":
    #rc.set_start_update(start, update, update_slow)
    rc.set_start_update(start, update, update_slow)
    rc.go()
