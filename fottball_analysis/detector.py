import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="fottball_analysis/models/best.pt", conf_threshold=0.5):
        """
        Initialize the YOLO model for object detection
        
        Args:
            model_path (str): Path to the YOLO model weights
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        
    def detect(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame to detect objects
            
        Returns:
            Dictionary containing detections for players, ball, referees
        """
        results = self.model(frame, verbose=False)[0]
        
        # Initialize empty lists for players, ball, and referees
        players = []
        ball = None
        referees = []
        
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = det
            
            if conf < self.conf_threshold:
                continue
                
            cls_id = int(cls_id)
            box = [int(x1), int(y1), int(x2), int(y2)]
            
            # Based on class IDs from your dataset
            # Adjust these based on your specific YOLO model's class mapping
            if cls_id == 0:  # Player
                players.append({'box': box, 'conf': conf, 'team': None})
            elif cls_id == 1:  # Ball
                ball = {'box': box, 'conf': conf}
            elif cls_id == 2:  # Referee
                referees.append({'box': box, 'conf': conf})
        
        return {
            'players': players,
            'ball': ball,
            'referees': referees
        }
    
    def get_center(self, box):
        """
        Get the center point of a bounding box
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Tuple (center_x, center_y)
        """
        x1, y1, x2, y2 = box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
    
    def get_width(self, box):
        """
        Get the width of a bounding box
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Width of the box
        """
        x1, _, x2, _ = box
        return x2 - x1