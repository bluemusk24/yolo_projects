import cv2
import numpy as np

class CameraMotionEstimator:
    def __init__(self, max_corners=100, quality_level=0.01, min_distance=30):
        """
        Initialize camera motion estimator
        
        Args:
            max_corners: Maximum number of corners to detect
            quality_level: Quality level for corner detection
            min_distance: Minimum distance between detected corners
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.prev_gray = None
        self.prev_points = None
        self.transformation_matrix = np.eye(3, 3, dtype=np.float32)
        self.cumulative_transform = np.eye(3, 3, dtype=np.float32)
        
    def estimate_motion(self, frame):
        """
        Estimate camera motion between frames
        
        Args:
            frame: Current frame
            
        Returns:
            Transformation matrix between frames
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            # Detect feature points
            self.prev_points = cv2.goodFeaturesToTrack(
                gray, 
                maxCorners=self.max_corners, 
                qualityLevel=self.quality_level, 
                minDistance=self.min_distance
            )
            return np.eye(3, 3, dtype=np.float32)
        
        # If we have previous points, track them
        if self.prev_points is not None and len(self.prev_points) > 0:
            # Calculate optical flow
            curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None
            )
            
            # Filter out points that couldn't be tracked
            if curr_points is not None:
                status = status.reshape(-1)
                prev_valid = self.prev_points[status == 1]
                curr_valid = curr_points[status == 1]
                
                if len(prev_valid) >= 4 and len(curr_valid) >= 4:
                    # Estimate rigid transformation
                    self.transformation_matrix, _ = cv2.estimateAffinePartial2D(
                        prev_valid, curr_valid
                    )
                    
                    # Add row to make it 3x3
                    if self.transformation_matrix is not None:
                        self.transformation_matrix = np.vstack([
                            self.transformation_matrix, [0, 0, 1]
                        ])
                    else:
                        # If estimation failed, use identity matrix
                        self.transformation_matrix = np.eye(3, 3, dtype=np.float32)
        
        # Update cumulative transformation
        self.cumulative_transform = self.cumulative_transform @ self.transformation_matrix
        
        # Update for next frame
        self.prev_gray = gray
        self.prev_points = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=self.max_corners, 
            qualityLevel=self.quality_level, 
            minDistance=self.min_distance
        )
        
        return self.transformation_matrix
    
    def adjust_point(self, point):
        """
        Adjust point coordinates based on camera motion
        
        Args:
            point: Original point (x, y)
            
        Returns:
            Adjusted point (x', y')
        """
        # Convert to homogeneous coordinates
        point_h = np.array([point[0], point[1], 1])
        
        # Apply inverse of cumulative transform to get "stable" coordinates
        inv_transform = np.linalg.inv(self.cumulative_transform)
        adjusted_point = inv_transform @ point_h
        
        return (int(adjusted_point[0]), int(adjusted_point[1]))
        
    def reset(self):
        """Reset camera motion estimator state"""
        self.prev_gray = None
        self.prev_points = None
        self.transformation_matrix = np.eye(3, 3, dtype=np.float32)
        self.cumulative_transform = np.eye(3, 3, dtype=np.float32)


class PerspectiveTransformer:
    def __init__(self):
        """Initialize perspective transformer"""
        self.transform_matrix = None
        self.field_width = 105  # Standard soccer field width in meters
        self.field_height = 68  # Standard soccer field height in meters
        
    def calibrate(self, src_points, field_corners=None):
        """
        Calibrate perspective transformation using known field points
        
        Args:
            src_points: Four corner points in the image
            field_corners: Four corners of the field in real-world coordinates
            
        Returns:
            Success flag
        """
        if field_corners is None:
            # Default to standard field corners (in meters)
            field_corners = np.array([
                [0, 0],  # Top-left
                [self.field_width, 0],  # Top-right
                [self.field_width, self.field_height],  # Bottom-right
                [0, self.field_height]  # Bottom-left
            ], dtype=np.float32)
        
        if len(src_points) != 4:
            return False
            
        src_points = np.array(src_points, dtype=np.float32)
        field_corners = np.array(field_corners, dtype=np.float32)
        
        # Calculate perspective transformation
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, field_corners)
        return True
        
    def transform_point(self, point):
        """
        Transform image point to real-world coordinates
        
        Args:
            point: Image point (x, y)
            
        Returns:
            Real-world coordinates (x', y') in meters
        """
        if self.transform_matrix is None:
            return None
            
        # Convert to homogeneous coordinates
        point_h = np.array([[[point[0], point[1]]]], dtype=np.float32)
        
        # Apply perspective transformation
        transformed = cv2.perspectiveTransform(point_h, self.transform_matrix)
        
        return (transformed[0][0][0], transformed[0][0][1])