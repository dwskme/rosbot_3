# Simple improvement to find_lane_center in utils.py

import numpy as np
import cv2
import pandas as pd

# -------------------------
# Postprocess model output
# -------------------------
def postprocess_mask(mask_tensor):
    mask_np = mask_tensor.squeeze().cpu().numpy()
    return (mask_np > 0.5).astype(np.uint8) * 255

# -------------------------
# Simple stable lane center with smoothing
# -------------------------
last_center = 128  # Global variable to remember last center

def find_lane_center(mask):
    global last_center
    
    rows = mask.shape[0]
    cols = mask.shape[1]
    
    # Check multiple rows instead of just middle
    check_rows = [rows//2, rows//2 + 10, rows//2 - 10]
    centers = []
    
    for row in check_rows:
        if 0 <= row < rows:
            nonzero = np.where(mask[row] > 0)[0]
            if len(nonzero) > 10:  # Need at least 10 pixels
                centers.append(np.mean(nonzero))
    
    if len(centers) == 0:
        return None  # No lanes detected
    
    # Calculate new center
    new_center = int(np.mean(centers))
    
    # Simple smoothing - don't jump too much
    max_jump = 30
    if abs(new_center - last_center) > max_jump:
        # Large jump - move slowly towards new center
        if new_center > last_center:
            new_center = last_center + max_jump
        else:
            new_center = last_center - max_jump
    
    # Simple exponential smoothing
    smoothed_center = int(0.7 * new_center + 0.3 * last_center)
    last_center = smoothed_center
    
    return smoothed_center

# -------------------------
# Basic PID controller
# -------------------------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# -------------------------
# Load odometry CSV
# -------------------------
def load_odometry_csv(path='data/odom_per_frame.csv'):
    df = pd.read_csv(path)
    df['Frame'] = df['Frame'].str.replace('.png', '', regex=False)
    df.set_index('Frame', inplace=True)
    return df

def get_odom_for_frame(df, frame_num):
    frame = f'frame_{frame_num:05d}'
    if frame in df.index:
        return df.loc[frame]
    return None
