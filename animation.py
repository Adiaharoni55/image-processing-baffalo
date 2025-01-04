import cv2
import numpy as np
import matplotlib.pyplot as plt
import rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


# Read the input image
input_image = cv2.imread('baffalo.png', cv2.IMREAD_GRAYSCALE)
if input_image is None:
    print("Error: Could not read image file")
    exit()

def process_rotated_image(image, angle):
    # Get image dimensions
    rows, cols = image.shape[:2]
    
    # Create rotation matrix and rotate image
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    
    # Process the rotated image
    enhanced = rectangle.clean_and_sharpen_image(rotated)
    edges = cv2.Canny(enhanced, 100, 200)
    lines = rectangle.find_lines_cv2(edges)
    merged = rectangle.merge_similar_lines(lines)
    
    # Detect corners
    corner_mask = rectangle.detect_and_mark_corners(enhanced)
    corners_lst = np.where((corner_mask == 255))
    corners_lst = list(set(zip(corners_lst[1], corners_lst[0])))
    
    # Find rectangles
    rec_lines, rec_intersections = rectangle.find_possible_rectangles_lines(merged, corners_lst, rotated)
    
    # Draw rectangles if found
    if rec_intersections:
        result = rectangle.draw_rectangle(rotated.copy(), rec_intersections)
    else:
        result = rotated.copy()
    
    return result

def animate(frame):
    plt.clf()
    angle = frame * 10  # Rotate by 10 degrees each frame
    
    # Process the rotated image
    result = process_rotated_image(input_image, angle)
    
    # Display the result
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.title(f'Rotation: {angle}°')

# Create the animation
fig = plt.figure(figsize=(10, 10))
anim = FuncAnimation(fig, animate, frames=36, interval=200, repeat=True)

# Save the animation (optional)
anim.save('rectangle_detection.gif', writer='pillow')

# Display the animation
plt.close()
HTML(anim.to_jshtml())