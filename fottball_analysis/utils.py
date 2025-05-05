import cv2
import numpy as np
import pickle
import os
from datetime import datetime

def crop_image(frame, box, padding=0):
    """
    Crop a region from an image based on bounding box
    
    Args:
        frame: Input frame
        box: Bounding box [x1, y1, x2, y2]
        padding: Optional padding around the box
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = box
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    
    return frame[y1:y2, x1:x2]

def save_data(data, filename):
    """
    Save data using pickle
    
    Args:
        data: Data to save
        filename: Output filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    """
    Load data using pickle
    
    Args:
        filename: Input filename
        
    Returns:
        Loaded data
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_current_timestamp():
    """
    Get current timestamp string
    
    Returns:
        Formatted timestamp string
    """
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def interpolate_ball_positions(positions_history, max_gap=5):
    """
    Interpolate missing ball positions
    
    Args:
        positions_history: List of ball positions (can include None)
        max_gap: Maximum gap size to interpolate
        
    Returns:
        Interpolated ball positions list
    """
    import pandas as pd
    
    # Convert to pandas Series
    pos_series = pd.Series(positions_history)
    
    # Cannot interpolate if not enough valid points
    valid_count = pos_series.notna().sum()
    if valid_count < 2:
        return positions_history
    
    # Interpolate missing values (limit to reasonable gaps)
    interpolated = pos_series.interpolate(method='linear', limit=max_gap)
    
    return interpolated.tolist()

def get_video_properties(video_path):
    """
    Get properties of a video file
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration': frame_count / fps if fps > 0 else 0
    }

def create_output_video_writer(input_video_path, output_path=None, codec='mp4v'):
    """
    Create a video writer with the same properties as the input video
    
    Args:
        input_video_path: Path to the input video
        output_path: Path for the output video (default: auto generate)
        codec: FourCC codec code
        
    Returns:
        VideoWriter object and output path
    """
    # Get video properties
    props = get_video_properties(input_video_path)
    
    if props is None:
        return None, None
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = get_current_timestamp()
        video_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_path = f"{video_name}_analyzed_{timestamp}.mp4"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        output_path, 
        fourcc, 
        props['fps'], 
        (props['width'], props['height'])
    )
    
    return writer, output_path

def draw_text_with_background(img, text, position, font_scale=1.0, 
                              font=cv2.FONT_HERSHEY_SIMPLEX, font_thickness=1,
                              text_color=(255, 255, 255), bg_color=(0, 0, 0),
                              padding=5):
    """
    Draw text with background rectangle
    
    Args:
        img: Input image
        text: Text to draw
        position: Position (x, y) for the text
        font_scale: Font scale
        font: Font type
        font_thickness: Thickness of the font
        text_color: Color of the text
        bg_color: Color of the background
        padding: Padding around the text
        
    Returns:
        Image with text
    """
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )
    
    # Calculate background rectangle dimensions
    bg_x1 = position[0] - padding
    bg_y1 = position[1] - text_height - padding
    bg_x2 = position[0] + text_width + padding
    bg_y2 = position[1] + padding
    
    # Draw background rectangle
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # Draw text
    cv2.putText(
        img, text, position, font, font_scale, text_color, font_thickness
    )
    
    return img