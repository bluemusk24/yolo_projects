import cv2
import numpy as np
from collections import defaultdict

class ObjectTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        Initialize the tracker for objects
        
        Args:
            max_disappeared (int): Maximum number of frames an object can be missing before deregistration
            max_distance (int): Maximum distance between centroids to consider it the same object
        """
        self.next_object_id = 0
        self.objects = {}  # Dictionary to store object IDs and their centroids
        self.disappeared = {}  # Dictionary to count frames an object has disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid):
        """
        Register a new object with a new ID
        
        Args:
            centroid: Center coordinates of the object
            
        Returns:
            ID assigned to the new object
        """
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        return object_id
        
    def deregister(self, object_id):
        """
        Deregister an object ID that has disappeared for too long
        
        Args:
            object_id: ID of the object to deregister
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, centroids):
        """
        Update tracked objects with new detections
        
        Args:
            centroids: List of centroids for detected objects
            
        Returns:
            Dictionary mapping object IDs to their centroids
        """
        # If no centroids, mark all objects as disappeared
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Deregister if object has disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            return self.objects
            
        # If we are currently not tracking any objects, register all
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            # Get IDs and centroids of existing objects
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distances between existing objects and new centroids
            distances = np.zeros((len(object_centroids), len(centroids)))
            for i, existing_centroid in enumerate(object_centroids):
                for j, new_centroid in enumerate(centroids):
                    distances[i, j] = np.sqrt(
                        (existing_centroid[0] - new_centroid[0]) ** 2 +
                        (existing_centroid[1] - new_centroid[1]) ** 2
                    )
            
            # Find the minimum distance for each row and column
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            # Keep track of which rows and columns we've already examined
            used_rows = set()
            used_cols = set()
            
            # Loop through the matched indexes
            for (row, col) in zip(rows, cols):
                # If this row or column has been used, or distance is too large
                if row in used_rows or col in used_cols or distances[row, col] > self.max_distance:
                    continue
                    
                # Update the centroid for this object ID
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                
                # Mark as used
                used_rows.add(row)
                used_cols.add(col)
            
            # Check for unused rows and columns
            unused_rows = set(range(distances.shape[0])) - used_rows
            unused_cols = set(range(distances.shape[1])) - used_cols
            
            # If there are more objects than centroids, check for disappearances
            if distances.shape[0] >= distances.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    # Deregister if object has disappeared for too long
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # Register new objects for unused centroids
                for col in unused_cols:
                    self.register(centroids[col])
                    
        return self.objects


class PlayerTracker(ObjectTracker):
    """Extended tracker for players with team assignment capabilities"""
    
    def __init__(self, max_disappeared=30, max_distance=50):
        super().__init__(max_disappeared, max_distance)
        self.team_assignments = {}  # Maps object_id to team (0 or 1)
        
    def register(self, centroid, team=None):
        """
        Register a new player with team assignment
        
        Args:
            centroid: Center coordinates of the player
            team: Team assignment (0 or 1)
            
        Returns:
            ID assigned to the new player
        """
        object_id = super().register(centroid)
        self.team_assignments[object_id] = team
        return object_id
        
    def deregister(self, object_id):
        """
        Deregister a player
        
        Args:
            object_id: ID of the player to deregister
        """
        super().deregister(object_id)
        if object_id in self.team_assignments:
            del self.team_assignments[object_id]
            
    def update_team(self, object_id, team):
        """
        Update team assignment for a player
        
        Args:
            object_id: ID of the player
            team: Team assignment (0 or 1)
        """
        if object_id in self.team_assignments:
            self.team_assignments[object_id] = team


class BallTracker(ObjectTracker):
    """Specialized tracker for the ball with additional handling for occlusions"""
    
    def __init__(self, max_disappeared=10, max_distance=30):
        super().__init__(max_disappeared, max_distance)
        self.positions_history = []  # Store historical positions for interpolation
        
    def update(self, centroids):
        """
        Update ball tracking with special handling for the single ball
        
        Args:
            centroids: List containing the ball centroid (or empty if not detected)
            
        Returns:
            Dictionary with the ball ID and centroid
        """
        objects = super().update(centroids)
        
        # Store positions for interpolation (including None if not detected)
        if len(objects) > 0:
            self.positions_history.append(list(objects.values())[0])
        else:
            self.positions_history.append(None)
            
        # Keep only the last 60 frames (2 seconds at 30fps)
        if len(self.positions_history) > 60:
            self.positions_history.pop(0)
            
        return objects
    
    def get_interpolated_position(self):
        """
        Get interpolated ball position when it's not detected
        
        Returns:
            Interpolated position or None if cannot interpolate
        """
        if len(self.objects) > 0:
            return list(self.objects.values())[0]
            
        # Try to interpolate from history
        valid_positions = [(i, pos) for i, pos in enumerate(self.positions_history) if pos is not None]
        
        if len(valid_positions) < 2:
            return None
            
        # Simple linear interpolation between the last two known positions
        last_idx, last_pos = valid_positions[-1]
        prev_idx, prev_pos = valid_positions[-2]
        
        # If the last known position is too old, don't interpolate
        if len(self.positions_history) - last_idx > 15:  # More than 0.5 seconds ago
            return None
            
        dx = (last_pos[0] - prev_pos[0]) / (last_idx - prev_idx)
        dy = (last_pos[1] - prev_pos[1]) / (last_idx - prev_idx)
        
        current_idx = len(self.positions_history) - 1
        steps = current_idx - last_idx
        
        # Project the position
        projected_x = last_pos[0] + dx * steps
        projected_y = last_pos[1] + dy * steps
        
        return (int(projected_x), int(projected_y))