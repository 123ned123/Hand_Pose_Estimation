import cv2
import numpy as np

# Connections between 21 MediaPipe-style hand joints
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),      # thumb
    (0,5),(5,6),(6,7),(7,8),      # index
    (0,9),(9,10),(10,11),(11,12), # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20) # pinky
]


def draw_skeleton(image, joints_2d, color=(0, 255, 0)):
    for i, j in HAND_CONNECTIONS:
        pt1 = tuple(joints_2d[i].astype(int))
        pt2 = tuple(joints_2d[j].astype(int))
        cv2.line(image, pt1, pt2, color, 2)
    for pt in joints_2d:
        cv2.circle(image, tuple(pt.astype(int)), 4, (0, 0, 255), -1)
    return image
