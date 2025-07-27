#!/usr/bin/env python
import cv2
import numpy as np

def clean_mask(raw_mask, thresh=0.5):
    """
    Threshold to binary → morphological close then open
    to remove speckles and fill gaps.
    """
    mask = (raw_mask > thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask

def get_lane_center(mask):
    """
    Compute the centroid x‑coordinate of the white mask
    in the bottom 20% of the image.
    """
    h, w = mask.shape
    bottom = mask[int(h * 0.8):, :]
    m = cv2.moments(bottom)
    if m['m00'] > 0:
        return int(m['m10'] / m['m00'])
    else:
        return w // 2

def hough_lines(mask, 
                rho=1, theta=np.pi/180, threshold=50,
                min_line_len=50, max_line_gap=10):
    """
    (Optional) Detect line segments via Probabilistic Hough.
    Returns a list of lines [[x1,y1,x2,y2], ...].
    """
    return cv2.HoughLinesP(mask, rho, theta, threshold,
                           minLineLength=min_line_len,
                           maxLineGap=max_line_gap)
