import cv2
import numpy as np
import matplotlib.pyplot as plt
import math



def detect_and_mark_corners(source):
    gray = cv2.threshold(source, 130, 255, cv2.THRESH_BINARY)[1]

    corners_scale = cv2.cornerHarris(gray, 3, 3, 0.05)
    corners_scale = cv2.dilate(corners_scale, None)

    corners_normalized = cv2.normalize(corners_scale, None, 0, 1, cv2.NORM_MINMAX)

    corners = (corners_normalized * 255).astype(np.uint8)
    corners = cv2.threshold(corners, np.mean(corners), 255, cv2.THRESH_BINARY)[1]
    
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(corners, kernel, iterations=1)


def find_lines_through_rectangle(lines, corners, tolerance=2.0):
    def point_to_line_distance(point, rho, theta):
        """Calculate distance from a point to a line in rho-theta form"""
        x, y = point
        # Distance formula: |rho - x*cos(theta) - y*sin(theta)|
        return abs(rho - x * np.cos(theta) - y * np.sin(theta))

    # Extract corner points
    top_left = corners['top_left']
    top_right = corners['top_right']
    bottom_left = corners['bottom_left']
    bottom_right = corners['bottom_right']
    
    # For each point, find lines that pass near it
    point_to_lines = {
        'top': [],     # Lines near top edge points
        'bottom': [],  # Lines near bottom edge points
        'left': [],    # Lines near left edge points
        'right': []    # Lines near right edge points
    }
    
    # Check each line
    for i, (rho, theta) in enumerate(lines):
        # Check distances to all points
        d_tl = point_to_line_distance(top_left, rho, theta)
        d_tr = point_to_line_distance(top_right, rho, theta)
        d_bl = point_to_line_distance(bottom_left, rho, theta)
        d_br = point_to_line_distance(bottom_right, rho, theta)
        
        # Horizontal line check (top)
        if d_tl < tolerance and d_tr < tolerance:
            point_to_lines['top'].append((i, rho, theta))
        
        # Horizontal line check (bottom)
        if d_bl < tolerance and d_br < tolerance:
            point_to_lines['bottom'].append((i, rho, theta))
            
        # Vertical line check (left)
        if d_tl < tolerance and d_bl < tolerance:
            point_to_lines['left'].append((i, rho, theta))
            
        # Vertical line check (right)
        if d_tr < tolerance and d_br < tolerance:
            point_to_lines['right'].append((i, rho, theta))

    # Print all found lines
    print("\nAll lines found near edges:")
    for edge, edge_lines in point_to_lines.items():
        print(f"\n{edge.capitalize()} edge: {len(edge_lines)} lines")
        for idx, rho, theta in edge_lines:
            angle_degrees = np.rad2deg(theta)
            print(f"  Line {idx}: rho={rho:.2f}, theta={theta:.2f} rad ({angle_degrees:.2f} degrees)")
    
    # Return all found lines if we have at least one line for each edge
    if all(len(lines) > 0 for lines in point_to_lines.values()):
        return {
            'top': [line for line in point_to_lines['top']],
            'bottom': [line for line in point_to_lines['bottom']],
            'left': [line for line in point_to_lines['left']],
            'right': [line for line in point_to_lines['right']]
        }
    else:
        return None

def draw_lines_from_hough(img, lines, color=(255, 0, 0), thickness=2):
    """
    Draw lines on an image given their rho and theta values from Hough transform.
    
    Parameters:
    img: Input image
    lines: List of (rho, theta) pairs
    color: Line color in BGR format (default: red)
    thickness: Line thickness (default: 2)
    
    Returns:
    img_with_lines: Image with drawn lines
    """
    # Make a copy of the image to draw on
    img_with_lines = img.copy()
    height, width = img.shape[:2]
    
    # Convert each line from (rho, theta) to endpoint coordinates
    for rho, theta in lines:
        # Prevent division by zero
        if math.sin(theta) == 0:
            continue
            
        # Calculate the two points on the line
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # Calculate line endpoints
        # These points are chosen far enough to cross the image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        # Draw the line
        cv2.line(img_with_lines, (x1, y1), (x2, y2), color, thickness)
    
    return img_with_lines

def clean_and_sharpen_image(img, brightness_adjustments=[100, 70]):
    """
    Enhance image clarity through multiple passes of brightness adjustment and noise reduction.

    Args:
        img: Input image array
        brightness_adjustments: List of brightness values to apply sequentially
        filter_passes: Number of noise reduction filter passes

    Returns:
        filtered_img: Enhanced image array with reduced noise and adjusted brightness
    """


    def apply_noise_reduction_sequence(noisy_img):
        """Apply progressive noise reduction with decreasing filter strength."""
        filtered_img = noisy_img.copy()
        for strength in range(1, 9):
            filtered_img = normalize_image_range(filtered_img)
            filtered_img = cv2.bilateralFilter(
                filtered_img,
                9 - strength,
                (9 - strength) * 2,
                (9 - strength) * 2
            )
        return filtered_img


    def shift_brightness(image, offset):
        """
        Shifts the brightness of an image by adding/subtracting a constant value.

        Args:
            image: numpy array of the input image
            offset: brightness adjustment value (-255 to 255)

        Returns:
            numpy array with adjusted brightness, maintaining the original data type
        """
        array_float = image.astype(float)
        clipped = np.clip(array_float + offset, 0, 255)
        return clipped.astype(image.dtype)

    def scale_contrast(image, scale_factor):
        """
        Adjusts image contrast by multiplying pixel values by a scale factor.

        Args:
            image: numpy array of the input image
            scale_factor: multiplication factor for contrast adjustment

        Returns:
            numpy array with adjusted contrast, converted to uint8
        """
        contrast_image = image * scale_factor
        clipped = np.clip(contrast_image, 0, 255)
        return np.uint8(clipped)

    def normalize_image_range(image):
        """
        Normalizes image to use full intensity range (0-255) by adjusting
        brightness and contrast based on min/max values.

        Args:
            image: input grayscale image

        Returns:
            normalized image utilizing full intensity range
        """
        matrix = np.array(image)
        min_val = int(np.min(matrix))
        max_val = int(np.max(matrix))

        # Shift minimum to zero
        shifted = shift_brightness(image, -min_val)
        # Scale to use full range
        return scale_contrast(shifted, round(255 / (max_val - min_val), 2))


    # Initial cleanup
    filtered_img = np.uint8(np.clip(img, 0, 255))

    # Apply filtering passes
    for brightness in brightness_adjustments:
        filtered_img = shift_brightness(filtered_img, brightness)
        filtered_img = cv2.GaussianBlur(filtered_img, (3, 3), 0)
        filtered_img = apply_noise_reduction_sequence(filtered_img)

    # Final pass
    filtered_img = cv2.GaussianBlur(filtered_img, (3, 3), 0)
    filtered_img = apply_noise_reduction_sequence(filtered_img)
    return np.uint8(np.clip(filtered_img, 0, 255))

def find_lines_cv2(img_edges, min_threshold_ratio=0.1, max_threshold_ratio=0.25):
    # Apply HoughLines with optimal parameters
    lines = cv2.HoughLines(img_edges, rho=1, theta=np.pi/180, threshold=35)
    
    if lines is None:
        return []
    
    # Convert lines to your format and get accumulator values
    lines_with_votes = []
    for line in lines:
        rho, theta = line[0]
        
        # Create a mask for this line
        mask = np.zeros_like(img_edges)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # Get points along the line
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        # Draw the line
        cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        
        # Count votes by checking intersection with edge image
        votes = cv2.countNonZero(cv2.bitwise_and(img_edges, mask))
        lines_with_votes.append((rho, theta, votes))
    
    # Find maximum votes
    if not lines_with_votes:
        return []
        
    max_votes = max(vote for _, _, vote in lines_with_votes)
    
    # Filter lines based on vote thresholds
    threshold_min = max_votes * min_threshold_ratio
    threshold_max = max_votes * max_threshold_ratio
    
    # Filter lines within the threshold range
    filtered_lines = [(rho, theta) for rho, theta, votes in lines_with_votes 
                     if threshold_min < votes < threshold_max]
    
    return filtered_lines

def merge_similar_lines(lines, theta_threshold=0.05, row_threshold=1):
    if not lines:
        return []
    
    # Sort lines by theta (angle)
    sorted_by_theta = sorted(lines, key=lambda x: x[1])
    
    # Keep track of which lines have been merged
    used_lines = set()
    merged_lines = []
    
    for i in range(len(sorted_by_theta)):
        if i in used_lines:
            continue
            
        current_group = [sorted_by_theta[i]]
        used_lines.add(i)
        
        curr_row = sorted_by_theta[i][0]
        curr_theta = sorted_by_theta[i][1]
        
        # Find similar lines
        for j in range(i + 1, len(sorted_by_theta)):
            if j in used_lines:
                continue
                
            next_line = sorted_by_theta[j]
            
            # Check if the line is within both row and theta thresholds
            if (curr_row - row_threshold <= next_line[0] <= curr_row + row_threshold and 
                curr_theta - theta_threshold <= next_line[1] <= curr_theta + theta_threshold):
                current_group.append(next_line)
                used_lines.add(j)
        
        # Calculate averages for the group
        avg_row = sum(line[0] for line in current_group) / len(current_group)
        avg_theta = sum(line[1] for line in current_group) / len(current_group)
        
        merged_lines.append((avg_row, avg_theta))
    
    return merged_lines

def is_parallel(theta1, theta2, threshold_degrees=17):
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    
    diff = abs(theta1_deg - theta2_deg)
    same_dir = diff <= threshold_degrees
    opp_dir = abs(diff - 180) <= threshold_degrees
    
    return same_dir or opp_dir

def is_perpendicular(line1, line2, shape_matrix, window_size=20):
    """
    Debug version that prints detailed information about each check.
    """
    import numpy as np
    
    def get_intersection(rho1, theta1, rho2, theta2):        
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([rho1, rho2])
        
        try:
            x, y = np.linalg.solve(A, b)
            return (int(x), int(y))
        except np.linalg.LinAlgError:
            print("DEBUG: Lines are parallel - no intersection")
            return None
    
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    
    # Get intersection
    intersection = get_intersection(rho1, theta1, rho2, theta2)
    if intersection is None:
        return False, None
        
    x_int, y_int = intersection
    height, width = shape_matrix.shape
    
    # Check bounds
    if not (0 <= x_int < width and 0 <= y_int < height):
        return False, None
    
    # Create window
    x_min = max(0, x_int - window_size//2)
    x_max = min(width, x_int + window_size//2)
    y_min = max(0, y_int - window_size//2)
    y_max = min(height, y_int + window_size//2)
    
    
    def get_line_points(rho, theta, x_min, x_max, y_min, y_max):
        points = []
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        if abs(cos_t) > abs(sin_t):
            for y in range(y_min, y_max):
                x = int((rho - y * sin_t) / cos_t)
                if x_min <= x < x_max and shape_matrix[y, x]:
                    points.append((x, y))
        else:
            for x in range(x_min, x_max):
                y = int((rho - x * cos_t) / sin_t)
                if y_min <= y < y_max and shape_matrix[y, x]:
                    points.append((x, y))
        return points
    
    # Get points
    points1 = get_line_points(rho1, theta1, x_min, x_max, y_min, y_max)
    points2 = get_line_points(rho2, theta2, x_min, x_max, y_min, y_max)
    
    
    if len(points1) < 2 or len(points2) < 2:
        return False, intersection
    
    def get_local_direction(points):
        points = np.array(points)
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        return np.arctan2(dy, dx)
    
    # Calculate local angles
    local_theta1 = get_local_direction(points1)
    local_theta2 = get_local_direction(points2)
    
    
    # Check perpendicularity
    angle_diff = abs(local_theta1 - local_theta2) % np.pi
    # Change threshold from 0.2 radians to 15 degrees converted to radians
    threshold_degrees = 20
    threshold_radians = np.deg2rad(threshold_degrees)
    is_perp = abs(angle_diff - np.pi/2) <= threshold_radians
    
    return is_perp, intersection


def is_corner(intersection_point, corners, threshold=2):
    x, y = intersection_point
    for dx in range(-threshold, threshold + 1):
        for dy in range(-threshold, threshold + 1):
            if (x + dx, y + dy) in corners:
                return True
    return False


input_image = cv2.imread('baffalo.png', cv2.IMREAD_GRAYSCALE)
if input_image is None:
    print("Error: Could not read image file")
    exit()

rows, cols = input_image.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 50, 1)
input_image = cv2.warpAffine(input_image, M, (cols, rows))

enhanced_image = clean_and_sharpen_image(input_image)
edges = cv2.Canny(enhanced_image, 100, 200)

lines = find_lines_cv2(edges)

merged = merge_similar_lines(lines)


corner_mask = detect_and_mark_corners(enhanced_image)
corners_lst = np.where((corner_mask == 255))
corners_lst = list(set(zip(corners_lst[1], corners_lst[0]))) 

lines_to_chack = [merged[60], merged[115], merged[351], merged[350]]

# # Display results
# plt.imshow(edges, cmap='gray')
# plt.axis('off')
# plt.show()



# corners = {
#         'top_left': (120, 16),
#         'top_right': (129, 21),
#         'bottom_left': (106,54),
#         'bottom_right': (118, 58)
# }

# matching_lines = find_lines_through_rectangle(merged, corners, tolerance=2.0)


# if matching_lines:
#     lines_to_chack = [merged[325], merged[288], merged[11], merged[14]]
#     result = draw_lines_from_hough(input_image, lines_to_chack)
#     # is_perp, intersection = debug_perpendicularity_hough(lines_to_chack[0], lines_to_chack[2], input_image)
#     if not is_parallel(lines_to_chack[0][1], lines_to_chack[1][1]):
#         print("ERROR! NOT PERALLEL 0, 1", lines_to_chack[0], lines_to_chack[1])
#     if not is_parallel(lines_to_chack[2][1], lines_to_chack[3][1]):
#         print("ERROR! NOT PERALLEL 2, 3")
#     for i in [0, 1]:
#         for j in [2, 3]:
#             is_perp, intersection = is_perpendicular(lines_to_chack[i], lines_to_chack[j], input_image)
#             if not is_perp:
#                 print("ERROR! NOT PERPENDICULAR", i, j)
#             if not is_corner(intersection, corners_lst):
#                 print("ERROR! NOT CORNER", i, j)

#     # Display results
#     plt.imshow(result, cmap='gray')
#     plt.axis('off')
#     plt.show()


result = draw_lines_from_hough(input_image, lines_to_chack)

# Display results
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.show()
