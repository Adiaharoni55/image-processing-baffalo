import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def find_hough_lines(img_edges, min_threshold_ratio=0.1, max_threshold_ratio=0.25):
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

def detect_and_mark_corners(source):
    gray = cv2.threshold(source, 130, 255, cv2.THRESH_BINARY)[1]

    corners_scale = cv2.cornerHarris(gray, 3, 3, 0.05)
    corners_scale = cv2.dilate(corners_scale, None)

    corners_normalized = cv2.normalize(corners_scale, None, 0, 1, cv2.NORM_MINMAX)

    corners = (corners_normalized * 255).astype(np.uint8)
    corners = cv2.threshold(corners, np.mean(corners), 255, cv2.THRESH_BINARY)[1]
    
    kernel = np.ones((3, 3), np.uint8)
    reduce_noise = cv2.morphologyEx(corners, cv2.MORPH_OPEN, kernel)
    return cv2.erode(reduce_noise, kernel, iterations=1)

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

def get_line_points(x1, y1, x2, y2):
    """Get all integer points along a line using Bresenham's algorithm."""
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            
    points.append((x, y))
    return points

def is_parallel(theta1, theta2, threshold_degrees=17):
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    
    diff = abs(theta1_deg - theta2_deg)
    same_dir = diff <= threshold_degrees
    opp_dir = abs(diff - 180) <= threshold_degrees
    
    return same_dir or opp_dir

def is_perpendicular(line1, line2, shape_matrix, window_size=20):
    
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

def is_valid_edge(corner_set, edges_set, point1, point2, threshold=4):
    # Get all points along the line
    x1, y1 = point1
    x2, y2 = point2
    line_points = get_line_points(x1, y1, x2, y2)
    
    # Count corner points along the line (excluding endpoints)
    corner_count = 0
    edges_count = 0
    for point in line_points[1:-1]:  # Skip first and last points
        if point in corner_set:
            corner_count += 1
            if corner_count > threshold:
                return False, None
        if point not in edges_set:
            edges_count += 1
            if edges_count > threshold:
                return False, None

        
    return True, corner_count+edges_count

def find_possible_rectangles_lines(lines, corners_set, edges_set, img, min_width=9, max_width=14, min_height=34, max_height=44):
    def is_corner(intersection_point, corners, threshold=2):
        x, y = intersection_point
        for dx in range(-threshold, threshold + 1):
            for dy in range(-threshold, threshold + 1):
                if (x + dx, y + dy) in corners:
                    return True, (x + dx, y + dy)
        return False, None
    
    def calc_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    
    lines = set(lines)  # Convert to set for O(1) removal

    for top_line in lines:
        
        lines = lines - {top_line}
        remaining_lines = lines.copy()

        perpendicular_lines = []
        for perp in remaining_lines:
            is_prep, intersection = is_perpendicular(top_line, perp, img)
            if not is_prep:
                continue
            valid_corner, point = is_corner(intersection, corners_set)
            if valid_corner:
                perpendicular_lines.append((perp, point))
        
        
        # Early exit if not enough perpendicular lines
        if len(perpendicular_lines) < 2 or len(perpendicular_lines) > 50:
            continue

        is_width = True

        # Check all possible pairs of perpendicular lines
        n = len(perpendicular_lines)
        for i in range(n-1):
            left_line, top_left = perpendicular_lines[i]
            
            for j in range(i+1, n):
                right_line, top_right = perpendicular_lines[j]
                                
                if not is_parallel(left_line[1], right_line[1]):
                    continue

                top_line_distance = calc_distance(top_left, top_right)
                if not (min_width <= top_line_distance <= max_width): # maybe not eisth, could be height 
                    is_width = False
                
                    if not min_height <= top_line_distance <= max_height:
                        continue                    
                sum_bad_points = 0
                is_val, bad_points = is_valid_edge(corners_set, edges_set, top_left, top_right)
                if not is_val:
                    continue
                sum_bad_points += bad_points
                # Search for bottom line
                for bottom_line in remaining_lines - {left_line, right_line}:
                    
                    if not is_parallel(bottom_line[1], top_line[1]):
                        continue

                    is_perp1, bottom_left = is_perpendicular(bottom_line, left_line, img)
                    is_perp2, bottom_right = is_perpendicular(bottom_line, right_line, img)

                    if not is_perp1 or not is_perp2:
                        continue
                    
                    valid_bottom_right, bottom_right = is_corner(bottom_right, corners_set)
                    valid_bottom_left, bottom_left = is_corner(bottom_left, corners_set)
                    if not valid_bottom_right or not valid_bottom_left:
                        continue

                    left_line_distance = calc_distance(top_left, bottom_left)
                    right_line_distance = calc_distance(top_right, bottom_right)

                    if is_width:
                        if not (min_height <= left_line_distance <= max_height) or not (min_height <= right_line_distance <= max_height):
                            continue

                    else:
                        if not (min_width <= right_line_distance <= max_width) or not (min_width <= left_line_distance <= max_width):
                            continue

                    bottom_line_dis = calc_distance(bottom_left, bottom_right)
                    
                    if is_width:
                        if not min_width <= bottom_line_dis <= max_width:
                            continue
                    
                    else:
                        if not min_height <= bottom_line_dis <= max_height:
                            continue
                    
                    if not (350 <= top_line_distance*left_line_distance) <= 550 or not (350 <= top_line_distance*right_line_distance <= 550) or not (350 <= bottom_line_dis*right_line_distance <= 550) or not ((350 <= bottom_line_dis*left_line_distance <= 550)):
                        continue
                    
                    is_val, bad_points = is_valid_edge(corners_set, edges_set, bottom_left, top_left)
                    if not is_val:
                        continue
                    sum_bad_points += bad_points

                    is_val, bad_points = is_valid_edge(corners_set, edges_set, bottom_right, top_right)
                    if not is_val:
                        continue
                    sum_bad_points += bad_points


                    is_val, bad_points = is_valid_edge(corners_set, edges_set, bottom_left, bottom_right)
                    if not is_val:
                        continue
                    sum_bad_points += bad_points

                    if sum_bad_points <= 15:
                        return (
                            {
                                "top_line": top_line,
                                "bottom_line": bottom_line,
                                "right_line": right_line,
                                "left_line": left_line
                            },
                            {
                                "top_left": top_left,
                                "top_right": top_right,
                                "bottom_left": bottom_left,
                                "bottom_right": bottom_right
                            }
                        )
    return None, None


def main():
    input_image = cv2.imread('baffalo.png', cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print("Error: Could not read image file")
        exit()

    # rows, cols = input_image.shape[:2]
    # M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, 1)
    # rotated = cv2.warpAffine(input_image, M, (cols, rows))

    enhanced_image = clean_and_sharpen_image(input_image)
    edges = cv2.Canny(enhanced_image, 100, 200)
    lines = find_hough_lines(edges)
    merged = merge_similar_lines(lines)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_lst = np.where(edges_dilated == 255)
    edges_set = set(zip(edges_lst[1], edges_lst[0]))
    
    corner_mask = detect_and_mark_corners(enhanced_image)
    corners_lst = np.where((corner_mask == 255))
    corners_set = set(zip(corners_lst[1], corners_lst[0]))


    rec_lines, rec_intersections = find_possible_rectangles_lines(merged, corners_set, edges_set, input_image)


    if rec_intersections:
        rec_img = mark_corners(input_image, rec_intersections)
        plt.imshow(rec_img, cmap='gray')
        plt.axis('off')
        plt.show()

    else:
        print(f"no rectangles found in image")

if __name__ == "__main__":
    main()