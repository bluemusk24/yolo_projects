import cv2
import numpy as np
import pandas as pd
import time
from collections import defaultdict

class SpeedDistanceCalculator:
    def __init__(self, fps=30, pixels_per_meter=None):
        """
        Initialize speed and distance calculator
        
        Args:
            fps (float): Frames per second of the video
            pixels_per_meter (float): Conversion factor from pixels to meters
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter  # Can be None if using perspective transform
        self.positions = defaultdict(list)  # {player_id: [(frame_idx, x, y), ...]}
        self.last_frame_time = None
        self.current_frame_idx = 0
        self.speeds = {}  # {player_id: current_speed}
        self.total_distances = defaultdict(float)  # {player_id: total_distance}
        
    def update(self, player_positions, frame_idx=None, transformer=None):
        """
        Update player positions and calculate speeds
        
        Args:
            player_positions (dict): Dictionary of player IDs and their positions
            frame_idx (int): Current frame index
            transformer (PerspectiveTransformer): Optional transformer to convert to real-world coordinates
            
        Returns:
            Dictionaries of player speeds and cumulative distances
        """
        current_time = time.time()
        
        if frame_idx is None:
            frame_idx = self.current_frame_idx
            self.current_frame_idx += 1
        
        # Calculate time elapsed since last frame
        if self.last_frame_time is None:
            time_elapsed = 1.0 / self.fps  # Assume standard frame interval for first frame
        else:
            time_elapsed = current_time - self.last_frame_time
        
        self.last_frame_time = current_time
        
        # Update positions and calculate speeds
        for player_id, position in player_positions.items():
            # Transform position to real-world coordinates if transformer is provided
            real_world_pos = position
            if transformer is not None:
                real_world_pos = transformer.transform_point(position)
                if real_world_pos is None:
                    continue
            
            # Store position with frame index
            self.positions[player_id].append((frame_idx, *real_world_pos))
            
            # Calculate speed if we have at least two positions
            if len(self.positions[player_id]) >= 2:
                prev_frame_idx, prev_x, prev_y = self.positions[player_id][-2]
                curr_frame_idx, curr_x, curr_y = self.positions[player_id][-1]
                
                # Calculate distance in pixels or meters
                if transformer is not None:
                    # Already in meters if using transformer
                    distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                else:
                    # Convert pixels to meters if pixels_per_meter is provided
                    pixel_distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                    distance = pixel_distance / self.pixels_per_meter if self.pixels_per_meter else pixel_distance
                
                # Calculate frames elapsed
                frames_elapsed = curr_frame_idx - prev_frame_idx
                
                # Calculate time elapsed based on frames and FPS
                if frames_elapsed > 0:
                    time_elapsed = frames_elapsed / self.fps
                
                # Calculate speed in meters per second
                if time_elapsed > 0:
                    speed = distance / time_elapsed
                    self.speeds[player_id] = speed
                
                # Update total distance
                self.total_distances[player_id] += distance
        
        return self.speeds, self.total_distances
    
    def get_player_path(self, player_id, max_length=None):
        """
        Get the path of a player over time
        
        Args:
            player_id: ID of the player
            max_length: Maximum number of positions to return (most recent)
            
        Returns:
            List of (x, y) positions
        """
        positions = self.positions.get(player_id, [])
        
        if max_length is not None and len(positions) > max_length:
            positions = positions[-max_length:]
            
        # Extract just the x, y coordinates (not frame index)
        return [(x, y) for _, x, y in positions]
    
    def reset(self):
        """Reset all tracking data"""
        self.positions.clear()
        self.speeds.clear()
        self.total_distances.clear()
        self.last_frame_time = None
        self.current_frame_idx = 0


class BallPossessionAnalyzer:
    def __init__(self, possession_distance_threshold=2.0):
        """
        Initialize ball possession analyzer
        
        Args:
            possession_distance_threshold (float): Maximum distance in meters to consider a player in possession
        """
        self.possession_distance_threshold = possession_distance_threshold
        self.possession_history = []  # List of (frame_idx, team_idx, player_id)
        self.team_possession_frames = {0: 0, 1: 0}  # Frames per team with possession
        self.current_possession = None  # (team_idx, player_id) or None
        
    def update(self, ball_position, player_positions, player_teams, frame_idx, transformer=None):
        """
        Update ball possession analysis
        
        Args:
            ball_position: (x, y) position of the ball
            player_positions: Dictionary of player IDs and their positions
            player_teams: Dictionary of player IDs and their team indices
            frame_idx: Current frame index
            transformer: Optional perspective transformer
            
        Returns:
            Current possession status (team_idx, player_id) or None
        """
        if ball_position is None:
            # Ball not detected
            self.possession_history.append((frame_idx, None, None))
            return self.current_possession
        
        # Find the nearest player to the ball
        nearest_player_id = None
        nearest_distance = float('inf')
        
        for player_id, player_pos in player_positions.items():
            if player_id not in player_teams:
                continue
                
            # Calculate distance between player and ball
            if transformer is not None:
                # Transform to real-world coordinates
                player_real = transformer.transform_point(player_pos)
                ball_real = transformer.transform_point(ball_position)
                if player_real is None or ball_real is None:
                    continue
                distance = np.sqrt((player_real[0] - ball_real[0])**2 + (player_real[1] - ball_real[1])**2)
            else:
                # Use pixel distance
                distance = np.sqrt((player_pos[0] - ball_position[0])**2 + (player_pos[1] - ball_position[1])**2)
            
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_player_id = player_id
        
        # Check if any player is close enough to possess the ball
        if nearest_player_id is not None and nearest_distance < self.possession_distance_threshold:
            team_idx = player_teams[nearest_player_id]
            self.current_possession = (team_idx, nearest_player_id)
            self.team_possession_frames[team_idx] += 1
        else:
            self.current_possession = None
            
        # Record possession history
        if self.current_possession:
            team_idx, player_id = self.current_possession
            self.possession_history.append((frame_idx, team_idx, player_id))
        else:
            self.possession_history.append((frame_idx, None, None))
            
        return self.current_possession
    
    def get_possession_percentage(self):
        """
        Calculate possession percentage for each team
        
        Returns:
            Dictionary with team possession percentages
        """
        total_frames = sum(self.team_possession_frames.values())
        
        if total_frames == 0:
            return {0: 0.0, 1: 0.0}
            
        return {
            team: (frames / total_frames) * 100
            for team, frames in self.team_possession_frames.items()
        }
    
    def reset(self):
        """Reset possession analysis"""
        self.possession_history.clear()
        self.team_possession_frames = {0: 0, 1: 0}
        self.current_possession = None