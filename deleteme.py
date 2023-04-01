import cv2
import numpy as np

# Load the perspective image
img = cv2.imread('perspective_image.jpg')

# Define the corresponding points in the top-down view
src_pts = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=np.float32)

# Define the corresponding points in the perspective image
dst_pts = np.array([[50, 50], [20, 80], [80, 80], [50, 20]], dtype=np.float32)

# Compute the homography matrix
H, _ = cv2.findHomography(src_pts, dst_pts)

# Create an array of cone coordinates in the perspective image
cone_coords = np.array([[60, 70], [80, 90], [30, 50]], dtype=np.float32)

# Convert the coordinates to homogeneous coordinates
cone_coords_homog = np.hstack((cone_coords, np.ones((len(cone_coords), 1))))

# Apply the homography matrix
cone_coords_topdown_homog = np.matmul(H, cone_coords_homog.T).T

# Convert the coordinates back to (x, y) format
cone_coords_topdown = cone_coords_topdown_homog[:, :2] / cone_coords_topdown_homog[:, 2:]

# Display the results
print('Perspective image cone coordinates:', cone_coords)
print('Top-down view cone coordinates:', cone_coords_topdown)