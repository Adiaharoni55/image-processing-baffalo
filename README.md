# Rectangle Detection in the Image Baffalo 

This project implements an advanced computer vision system for detecting and marking rectangles in grayscale images. It uses various image processing techniques including edge detection, corner detection, and line detection to identify rectangular shapes with specific dimensions.

## Features

- Image enhancement and noise reduction
- Edge detection using Canny algorithm
- Corner detection using Harris corner detector
- Line detection using Hough transform
- Rectangle identification based on geometric constraints
- Visual marking of detected rectangles

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
   - Rectangle dimension verification

### Key Functions

- `clean_and_sharpen_image()`: Enhances image clarity through multiple processing passes
- `detect_and_mark_corners()`: Identifies and highlights corner points
- `find_hough_lines()`: Detects lines using the Hough transform
- `merge_similar_lines()`: Combines nearby parallel lines
- `find_possible_rectangles_lines()`: Identifies valid rectangles based on geometric constraints
- `mark_corners()`: Visualizes detected rectangle corners

### Parameters

The rectangle detection system looks for rectangles with the following constraints:
- Width: 9-14 pixels
- Height: 34-44 pixels
- Area: 350-550 square pixels
- Maximum corner point deviation: 2 pixels
- Maximum edge point deviation: 4 pixels

## Example Output

The script will display the input image with detected rectangles marked using red corner indicators. If no valid rectangles are found, it will print a notification message.

## Limitations

- Works only with grayscale images
- Optimized for specific rectangle dimensions
- Sensitive to image quality and noise
- May require parameter tuning for different image conditions

## Contributing

Feel free to submit issues and enhancement requests. Follow these steps to contribute:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
