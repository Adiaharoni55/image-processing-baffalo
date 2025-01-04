# 211749361, Adi Aharoni

import cv2
import numpy as np
import matplotlib.pyplot as plt
import HT_for_stud
import math
from itertools import combinations



# debug 
def get_rectangle_lines_from_indices(lines_list, matching_indices):
    """
    Extract the actual line parameters (rho, theta) from indices.
    
    Parameters:
    lines_list: list of (rho, theta) pairs representing all lines
    matching_indices: dictionary with {'top', 'bottom', 'left', 'right'} indices
    
    Returns:
    dictionary with the actual (rho, theta) values for each edge
    """
    rectangle_lines = {}
    for edge, index in matching_indices.items():
        rectangle_lines[edge] = lines_list[index]
    
    return rectangle_lines
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
    
    # # Print debug information
    # print("Found lines near edges:")
    # for edge, edge_lines in point_to_lines.items():
    #     print(f"{edge.capitalize()} edge: {len(edge_lines)} lines")
    #     for idx, rho, theta in edge_lines:
    #         print(f"  Line {idx}: rho={rho:.2f}, theta={theta:.2f}")
    
    # Check if we have lines for all edges
    if all(len(lines) > 0 for lines in point_to_lines.values()):
        return {
            'top': point_to_lines['top'][0][0],
            'bottom': point_to_lines['bottom'][0][0],
            'left': point_to_lines['left'][0][0],
            'right': point_to_lines['right'][0][0]
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
def find_line_intersection(rho1, theta1, rho2, theta2):
    """
    Find intersection point of two lines in rho-theta form.
    
    Parameters:
    rho1, theta1: parameters of first line
    rho2, theta2: parameters of second line
    
    Returns:
    (x, y) intersection point
    """
    # Create line equations matrix
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    
    b = np.array([rho1, rho2])
    
    # Solve the system of equations
    try:
        x, y = np.linalg.solve(A, b)
        return int(round(x)), int(round(y))
    except np.linalg.LinAlgError:
        return None
def get_rectangle_corners(rectangle_lines):
    """
    Find corners of rectangle from line parameters.
    
    Parameters:
    rectangle_lines: dictionary with 'top', 'bottom', 'left', 'right' lines in (rho, theta) form
    
    Returns:
    dictionary with corner coordinates
    """
    # Extract line parameters
    top_rho, top_theta = rectangle_lines['top']
    bottom_rho, bottom_theta = rectangle_lines['bottom']
    left_rho, left_theta = rectangle_lines['left']
    right_rho, right_theta = rectangle_lines['right']
    
    # Find intersections
    top_left = find_line_intersection(top_rho, top_theta, left_rho, left_theta)
    top_right = find_line_intersection(top_rho, top_theta, right_rho, right_theta)
    bottom_left = find_line_intersection(bottom_rho, bottom_theta, left_rho, left_theta)
    bottom_right = find_line_intersection(bottom_rho, bottom_theta, right_rho, right_theta)
    
    return {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }
def mark_corners(image, coordinates):
    """
    Mark the corners of a rectangle on an image with a custom arrow-like shape

    Args:
        image: Input grayscale image
        coordinates: Dictionary containing corner coordinates

    Returns:
        Image with marked corners in RGB format
    """
    # Convert grayscale to RGB while keeping grayscale look
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Define the marker pattern
    marker = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.uint8)

    # Get marker dimensions
    marker_height, marker_width = marker.shape
    half_height = marker_height // 2
    half_width = marker_width // 2

    # List of corner points
    corners = [
        coordinates['top_left'],
        coordinates['top_right'],
        coordinates['bottom_left'],
        coordinates['bottom_right']
    ]

    # Draw custom marker at each corner
    for corner in corners:
        x, y = corner

        # Calculate bounds for marker placement
        y_start = max(0, y - half_height)
        y_end = min(image.shape[0], y + half_height + 1)
        x_start = max(0, x - half_width)
        x_end = min(image.shape[1], x + half_width + 1)

        # Calculate marker array bounds
        marker_y_start = max(0, half_height - y)
        marker_y_end = marker_height - max(0, (y + half_height + 1) - image.shape[0])
        marker_x_start = max(0, half_width - x)
        marker_x_end = marker_width - max(0, (x + half_width + 1) - image.shape[1])

        # Apply marker where pattern is 1
        mask = marker[marker_y_start:marker_y_end, marker_x_start:marker_x_end] == 1
        result[y_start:y_end, x_start:x_end][mask] = [255, 0, 0]  # Red color

    return result


# proccess img
def find_edges(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    eroded = cv2.erode(image, kernel, iterations=1)
    image = dilated - eroded
    return cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]

def canny_proccess(image):
    img1 = HT_for_stud.gs_filter(image, 3)
    img2, D = HT_for_stud.gradient_intensity(img1)
    img2 = np.uint8(img2) 
    img2 = cv2.threshold(img2, 20, 255, cv2.THRESH_BINARY)[1]
    img3 = HT_for_stud.suppression(np.copy(img2), D)
    img4, weak = HT_for_stud.threshold(np.copy(img3), 20, 30)
    return HT_for_stud.tracking_queue(np.copy(img4), weak)

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

def clean_and_sharpen_image(img, brightness_adjustments=None):
    """
    Enhance image clarity through multiple passes of brightness adjustment and noise reduction.

    Args:
        img: Input image array
        brightness_adjustments: List of brightness values to apply sequentially
        filter_passes: Number of noise reduction filter passes

    Returns:
        filtered_img: Enhanced image array with reduced noise and adjusted brightness
    """

    if brightness_adjustments is None:
        brightness_adjustments = [100, 70]

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

# find lines and proccess them
def find_lines(accumulator, thetas, rhos, minthreshold, maxthreshold):
    lines = []
    for rho_idx in range(accumulator.shape[0]):
        for theta_idx in range(accumulator.shape[1]):
            if minthreshold < accumulator[rho_idx, theta_idx] < maxthreshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                lines.append((rho, theta))
    return lines

def merge_similar_lines(lines, theta_threshold=0.1, row_threshold=1):
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

def find_perpendicular_lines(lines, theta_threshold=0.05):
    if not lines:
        return []
        
    sorted_by_theta = sorted(lines, key=lambda x: x[1])
    parallel_groups = []
    i = 0
    
    # First, group parallel lines
    while i < len(sorted_by_theta):
        cur_arr = [sorted_by_theta[i]]
        cur_theta = sorted_by_theta[i][1]
        j = i + 1
        
        while j < len(sorted_by_theta):
            if abs(sorted_by_theta[j][1] - cur_theta) <= theta_threshold:
                cur_arr.append(sorted_by_theta[j])
                j += 1
            else:
                break
                
        parallel_groups.append(cur_arr)
        i = j

    # Function to check if two angles are perpendicular
    def is_perpendicular(theta1, theta2, threshold):
        angle_diff = abs(theta1 - theta2) % (np.pi/2)
        return abs(angle_diff - np.pi/2) <= threshold

    result_lines = []
    used_groups = set()
    
    for i, group1 in enumerate(parallel_groups):
        if i in used_groups:
            continue
            
        theta1 = group1[0][1]
        for j, group2 in enumerate(parallel_groups):
            if i != j and j not in used_groups:
                theta2 = group2[0][1]
                if is_perpendicular(theta1, theta2, theta_threshold):
                    # Add all lines from both groups to result
                    result_lines.extend(group1)
                    result_lines.extend(group2)
                    used_groups.add(i)
                    used_groups.add(j)
                    break

    return result_lines


def draw_rectangles(input_image, rectangles):
    output_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    
    for rect_info in rectangles:
        rect_lines = list(rect_info[0])
        height, width = rect_info[1], rect_info[2]
        
        rho1, theta1 = rect_lines[0]
        rho2, theta2 = rect_lines[2]
        
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        
        denominator = (a1*b2 - a2*b1)
        if abs(denominator) < 1e-10:
            continue
            
        x = (rho1*b2 - rho2*b1) / denominator
        y = (rho2*a1 - rho1*a2) / denominator
        
        if not (0 <= x < input_image.shape[1] and 0 <= y < input_image.shape[0]):
            continue
            
        angle = theta1  # Using theta1 as the rotation angle
        rect = ((int(x), int(y)), (height, width), np.degrees(angle))
        box = np.intp(cv2.boxPoints(rect))
        
        cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)
    
    return output_image

def check_rectangle_in_results(rectangles, target_corners):
    found_indices = []
    
    for i, rect in enumerate(rectangles):
        rect_lines = list(rect[0])
        rho1, theta1 = rect_lines[0]
        rho2, theta2 = rect_lines[2]
        
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        
        denominator = (a1*b2 - a2*b1)
        if abs(denominator) < 1e-10:
            continue
            
        x = (rho1*b2 - rho2*b1) / denominator
        y = (rho2*a1 - rho1*a2) / denominator
        
        # Calculate rectangle corners based on x, y, height, width, and angle
        height, width = rect[1], rect[2]
        angle = theta1
        box = np.intp(cv2.boxPoints(((int(x), int(y)), (height, width), np.degrees(angle))))
        
        # Compare with target corners
        tolerance = 6
        corners_match = all(
            any(abs(corner[0] - target[0]) <= tolerance and 
                abs(corner[1] - target[1]) <= tolerance 
                for corner in box)
            for target in [target_corners['top_left'], target_corners['top_right'],
                         target_corners['bottom_left'], target_corners['bottom_right']]
        )
        
        if corners_match:
            found_indices.append(i)
    
    return found_indices


def find_line_intersections(lines, img_shape):
    """
    Find intersections between lines given in Hesse normal form (rho, theta)
    and filter points within image boundaries.
    
    Args:
        lines: List of tuples (rho, theta) representing lines in Hesse normal form
        img_shape: Tuple of (height, width) representing image dimensions
        
    Returns:
        List of intersection points (x, y) that fall within image boundaries
    """
    intersections = []
    n = len(lines)
    
    # Convert lines from (rho, theta) to ax + by = c form
    # x cos(theta) + y sin(theta) = rho
    line_params = []
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        c = rho
        line_params.append((a, b, c))
    
    # Find intersections between all pairs of lines
    for i in range(n):
        for j in range(i + 1, n):
            a1, b1, c1 = line_params[i]
            a2, b2, c2 = line_params[j]
            
            # Check if lines are parallel (determinant = 0)
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-10:  # Small threshold for numerical stability
                continue
                
            # Solve system of equations
            x = (b2 * c1 - b1 * c2) / det
            y = (a1 * c2 - a2 * c1) / det
            
            # Check if intersection point is within image boundaries
            if (0 <= x < img_shape[1] and 0 <= y < img_shape[0]):
                intersections.append((x, y))
    
    return intersections

def draw_rectangle_on_image(img, rectangle_data):
    import cv2
    import numpy as np
    
    # Convert to BGR if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Get intersection points from lines
    lines = list(rectangle_data[0])
    intersections = []
    for i in range(len(lines)):
        rho1, theta1 = lines[i]
        for j in range(i + 1, len(lines)):
            rho2, theta2 = lines[j]
            
            # Calculate intersection
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            
            try:
                x, y = np.linalg.solve(A, b)
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    intersections.append((int(x[0]), int(y[0])))
            except:
                continue
    
    # Draw rectangle using intersection points
    if len(intersections) == 4:
        # Sort points to form rectangle
        pts = np.array(intersections[:4])
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        # Draw rectangle
        cv2.polylines(img, [np.int32(rect)], True, (0, 255, 0), 2)
        
    return img




def find_possible_rectangles_lines(img_shape, lines, min_width=10, max_width=15, min_height=37, max_height=42):
    lines = np.array(lines)
    parallel_threshold = 0.1
    perpendicular_threshold = 0.18
    res = set()
    
    # Vectorized angle differences computation
    angle_diffs = np.abs(lines[:, 1][:, np.newaxis] - lines[:, 1])
    
    # Create a mask for valid lines (initially all True)
    valid_lines_mask = np.ones(len(lines), dtype=bool)
    
    # Vectorized parallel and perpendicular line finding
    parallel_mask = angle_diffs <= parallel_threshold
    perpendicular_mask = np.abs(angle_diffs - np.pi/2) <= perpendicular_threshold
    
    for line1_idx in range(len(lines)):
        if not valid_lines_mask[line1_idx]:  # Skip if line already used
            continue
            
        # Only consider valid lines for parallel and perpendicular checks
        current_parallel_mask = parallel_mask[line1_idx] & valid_lines_mask
        current_perpendicular_mask = perpendicular_mask[line1_idx] & valid_lines_mask
        
        parallel_lines = np.where(current_parallel_mask)[0]
        perpendicular_lines = np.where(current_perpendicular_mask)[0]
        
        if len(parallel_lines) > 60 or len(perpendicular_lines) > 60:
            continue
            
        for line2_idx in parallel_lines:
            if line2_idx != line1_idx:
                for perp1_idx, perp2_idx in combinations(perpendicular_lines, 2):
                    intersection_points = find_line_intersections(
                        [lines[line1_idx], lines[line2_idx], lines[perp1_idx], lines[perp2_idx]], 
                        img_shape
                    )
                    if len(intersection_points) != 4:
                        continue
                        
                    points = np.array(intersection_points)
                    x_coords = points[:, 0]
                    width = np.ptp(x_coords)
                    if not (min_width <= width <= max_width):
                        continue

                    y_coords = points[:, 1]
                    height = np.ptp(y_coords)
                    if not (min_height <= height <= max_height) or not (500 <= (width * height) <= 550):
                        continue
                    
                    res.add((
                        frozenset([tuple(lines[line1_idx]), tuple(lines[line2_idx]), 
                                tuple(lines[perp1_idx]), tuple(lines[perp2_idx])]),
                                width, height))
                    
                    # Mark these lines as used
                    valid_lines_mask[[line1_idx, line2_idx, perp1_idx, perp2_idx]] = False

    return list(res)



def main():
    input_image = cv2.imread('baffalo.png', cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print("Error: Could not read image file")
        exit()

    # rows, cols = input_image.shape[:2]
    # M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    # input_image = cv2.warpAffine(input_image, M, (cols, rows))

    
    img_cpy = np.copy(input_image)
    
    enhanced_image = clean_and_sharpen_image(
        img_cpy,
        brightness_adjustments=[100, 70],
    )
    
    # Ensure enhanced_image is uint8 before finding edges
    enhanced_image = np.uint8(np.clip(enhanced_image, 0, 255))

    img_edges = find_edges(enhanced_image)

    accumulator, thetas, rhos = HT_for_stud.hough_line(img_edges)

    threshold_min = np.max(accumulator) * 0.17
    threshold_max = np.max(accumulator) * 0.19
    # # After running your find_lines function:
    lines = find_lines(accumulator, thetas, rhos, threshold_min, threshold_max)
    merged = merge_similar_lines(lines)
    all_lines = find_perpendicular_lines(merged)

    rectangles = find_possible_rectangles_lines(input_image.shape, all_lines)
    print(len(rectangles))
    # input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    result = draw_rectangle_on_image(input_image, rectangles[0])

    for i in range(1, len(rectangles)):
        result = draw_rectangle_on_image(result, rectangles[i])

    # Display results
    plt.imshow(result)
    plt.axis('off')
    plt.show()

    # output_image = draw_rectangles(input_image, [rectangles[18173]] )

    # # Convert BGR to RGB for matplotlib
    # output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    # plt.imshow(output_image_rgb)
    # plt.title('Rectangle')
    # plt.axis('off')  # Optional: hide axes
    # plt.show()

    # output_image = draw_rectangles(input_image, rectangles)
    # cv2.imwrite('output.png', output_image)    
    
    # for p in possible_rectangles_lines:
    #     print(p)

    # print(len(all_lines))

    # # Your rectangle coordinates
    # corners = {
    #     'top_left': (252, 167),
    #     'top_right': (263, 167),
    #     'bottom_left': (252, 206),
    #     'bottom_right': (263, 206)
    # }

    # # print(check_rectangle_in_results(rectangles, corners))
    # # # Find lines that form the rectangle
    # matching_lines = find_lines_through_rectangle(all_lines, corners, tolerance=2.0)
    # print(matching_lines)
    # # rectangle_lines = get_rectangle_lines_from_indices(all_lines, matching_lines)
    
    # # corners = get_rectangle_corners(rectangle_lines)
    # # result = mark_corners(input_image, corners)
    # result = draw_lines_from_hough(input_image, [all_lines[0], all_lines[6], all_lines[33], all_lines[411]])

    # # Display results
    # plt.imshow(result, cmap='gray')
    # plt.axis('off')
    # plt.show()


if __name__ == "__main__":
    main()
