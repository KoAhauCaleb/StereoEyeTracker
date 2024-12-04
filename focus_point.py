#import screen_calibration
import numpy as np

def closest_point_between_lines(P1, D1, P2, D2):
    # Normalize direction vectors
    D1 = D1 / np.linalg.norm(D1)
    D2 = D2 / np.linalg.norm(D2)

    # Get the cross product of the direction vectors
    cross_product = np.cross(D1, D2)
    dot_product = np.dot(cross_product, cross_product)

    # Make sure the lines are not parallel
    if dot_product < 1e-6:
        return (P1 + P2) / 2
    
    # Calculate the closest points
    t1 = np.dot(np.subtract(P2, P1), np.cross(D2, np.cross(D1, D2))) / dot_product
    t2 = np.dot(np.subtract(P1, P2), np.cross(D1, np.cross(D2, D1))) / dot_product

    closest_1 = P1 + (t1 * D1)
    closest_2 = P2 + (t2 * D2)
    
    # return the midpoint between the closest points
    return (closest_1 + closest_2) / 2

P_left = np.array([0,0,0]) # placeholder for the left eye position
P_right = np.array([0.06,0,0]) # placeholder for the right eye position
D_left = np.array([0.2,0.1,1]) # placeholder for the left eye vector
D_right = np.array([0.18,0.12,1]) # placeholder for the right eye vector
midpoint = closest_point_between_lines(P_left, D_left, P_right, D_right)

print(midpoint)

#calibration_points = screen_calibration.get_calibration_points()