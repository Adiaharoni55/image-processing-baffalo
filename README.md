# Rectangle Detection in the Image Baffalo 

This project implements an advanced computer vision system for detecting and marking rectangles in grayscale images. It uses various image processing techniques including edge detection, corner detection, and line detection to identify rectangular shapes based on their aspect ratio.

## Features

- Image enhancement and noise reduction
- Edge detection using Canny algorithm
- Corner detection using Harris corner detector
- Line detection using Hough transform
- Rectangle identification based on geometric constraints and aspect ratio
- Visual marking of detected rectangles
- Rotation-invariant detection (0-360 degrees)

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib

Install required packages using:
```bash
pip install opencv-python numpy matplotlib
```

## Usage

1. Import the project:
```python
from rectangle_detector import main
```

2. Run the detection on an image:
```python
main()
```

The script will:
- Load the target image
- Process and enhance the image
- Detect edges, corners, and lines
- Identify rectangles matching specified criteria
- Display the results with marked corners

## Technical Details

### Image Processing Pipeline

1. **Image Enhancement**
   - Brightness adjustment
   - Noise reduction using bilateral filtering
   - Contrast normalization

2. **Feature Detection**
   - Edge detection using Canny algorithm
   - Corner detection using Harris corner detector
   - Line detection using Hough transform

3. **Rectangle Detection**
   - Line merging and filtering
   - Parallel and perpendicular line detection
   - Corner validation
   - Rectangle dimension verification based on aspect ratio

### Key Functions

- `clean_and_sharpen_image()`: Enhances image clarity through multiple processing passes
- `detect_and_mark_corners()`: Identifies and highlights corner points
- `find_hough_lines()`: Detects lines using the Hough transform
- `merge_similar_lines()`: Combines nearby parallel lines
- `find_possible_rectangles_lines()`: Identifies valid rectangles based on geometric constraints and aspect ratio
- `mark_corners()`: Visualizes detected rectangle corners


## Limitations

- Works only with grayscale images
- Requires minimum side length of 9 pixels
- Optimized for rectangles with approximately 1:4 aspect ratio
- Sensitive to image quality and noise
- May require parameter tuning for different image conditions

### Rotation Invariance

The system has been tested and verified to work across all angles (0-360 degrees). A demonstration GIF (rectangle_detection.gif) shows the detection process at 10-degree intervals, confirming the algorithm's ability to identify rectangles regardless of orientation.

## Example Output

The script will display the input image with detected rectangles marked using red corner indicators. If no valid rectangles are found, it will print a notification message.

