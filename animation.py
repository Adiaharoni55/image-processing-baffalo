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
    
    try:
        # Process the rotated image
        enhanced = rectangle.clean_and_sharpen_image(rotated)
        edges = cv2.Canny(enhanced, 100, 200)
        lines = rectangle.find_hough_lines(edges)
        merged = rectangle.merge_similar_lines(lines)
        
        # detect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        edges_lst = np.where(edges_dilated == 255)
        edges_set = set(zip(edges_lst[1], edges_lst[0]))

        # detect corners
        corner_mask = rectangle.detect_and_mark_corners(enhanced)
        corners_lst = np.where((corner_mask == 255))
        corners_set = set(zip(corners_lst[1], corners_lst[0]))
        
        # Find rectangles - passing rotated image as both source and destination
        rec_lines, rec_intersections = rectangle.find_possible_rectangles_lines(merged, corners_set, edges_set, rotated)  # Changed from input_image to rotated
        
        # Draw rectangles if found
        if rec_intersections:
            result = rectangle.mark_corners(rotated.copy(), rec_intersections)
            print(f"Found rectangle at angle {angle}°")
        else:
            result = rotated.copy()
            
    except Exception as e:
        print(f"Error processing frame at angle {angle}: {str(e)}")
        result = rotated.copy()
    
    return result  # Added return statement

def animate(frame):
    angle = frame * 10  # Rotate by 10 degrees each frame
    
    # Process the rotated image
    result = process_rotated_image(input_image, angle)
    
    # Clear previous plot
    plt.clf()
    
    # Display the result
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.title(f'Rotation: {angle}°')
    
    # Return the figure for the animation
    return plt.gca()

# Set up the figure
fig = plt.figure(figsize=(10, 10))

# Create the animation with error handling
try:
    anim = FuncAnimation(fig, animate, frames=36, interval=200, repeat=True)
    anim.save('rectangle_detection.gif', writer='pillow')
    plt.close()
except Exception as e:
    print(f"Error creating animation: {str(e)}")
    plt.close()