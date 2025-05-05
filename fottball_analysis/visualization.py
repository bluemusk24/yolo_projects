import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        """Initialize the visualization module"""
        # Default team colors (will be updated with actual team colors)
        self.team_colors = {
            0: (0, 0, 255),    # Red for team 0
            1: (255, 0, 0),    # Blue for team 1
            None: (0, 255, 0)  # Green for unclassified
        }
        self.ball_color = (0, 165, 255)  # Orange
        self.referee_color = (0, 0, 0)   # Black
        
    def set_team_colors(self, team_colors):
        """
        Set team colors for visualization
        
        Args:
            team_colors: Dictionary mapping team indices to BGR colors
        """
        self.team_colors.update(team_colors)
        
    def draw_elliptical_bounding_boxes(self, frame, detections, player_teams=None, player_ids=None):
        """
        Draw elliptical bounding boxes for players, ball, and referees
        
        Args:
            frame: Input frame
            detections: Detection results from ObjectDetector
            player_teams: Dictionary mapping player indices to team indices
            player_ids: Dictionary mapping object IDs to player indices
            
        Returns:
            Frame with elliptical bounding boxes
        """
        vis_frame = frame.copy()
        
        # Draw players
        if 'players' in detections:
            for i, player in enumerate(detections['players']):
                box = player['box']
                team = None
                
                # Get team assignment if available
                if player_teams is not None and i in player_teams:
                    team = player_teams[i]
                
                # Get player ID if available
                player_id = None
                if player_ids is not None and i in player_ids:
                    player_id = player_ids[i]
                
                # Calculate the center, axes, and angle for the ellipse
                center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))  # Center of the box
                axes = (int((box[2] - box[0]) / 2), int((box[3] - box[1]) / 2))  # Axes of the ellipse
                angle = 0  # You can add rotation logic if needed

                # Get the color for the ellipse based on the team
                color = self.team_colors.get(team, (255, 255, 255))  # Default to white if no team
                
                # Draw the elliptical bounding box
                cv2.ellipse(vis_frame, center, axes, angle, 0, 360, color, 2)
                
                # Draw player ID
                if player_id is not None:
                    cv2.putText(vis_frame, f"ID: {player_id}", (box[0], box[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ball
        if 'ball' in detections and detections['ball'] is not None:
            ball_box = detections['ball']['box']
            # Draw circle instead of box for ball
            center_x = (ball_box[0] + ball_box[2]) // 2
            center_y = (ball_box[1] + ball_box[3]) // 2
            radius = max(3, (ball_box[2] - ball_box[0]) // 2)
            cv2.circle(vis_frame, (center_x, center_y), radius, self.ball_color, -1)
            
            # Draw triangle pointer on top of the ball
            self.draw_triangle(vis_frame, (center_x, center_y - radius - 10), 10, self.ball_color)
        
        # Draw referees
        if 'referees' in detections:
            for ref in detections['referees']:
                box = ref['box']
                cv2.rectangle(vis_frame, (box[0], box[1]), (box[2], box[3]), self.referee_color, 2)
                cv2.putText(vis_frame, "Ref", (box[0], box[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.referee_color, 2)
        
        return vis_frame
    
    def draw_triangle(self, frame, position, size, color):
        """
        Draw a triangle pointing up
        
        Args:
            frame: Input frame
            position: Position of the triangle tip (x, y)
            size: Size of the triangle
            color: Color of the triangle
        """
        x, y = position
        pts = np.array([ 
            [x, y], 
            [x - size, y + size], 
            [x + size, y + size] 
        ], np.int32)
        
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], color)
    
    def draw_player_paths(self, frame, player_paths, player_teams=None, path_length=20):
        """
        Draw player movement paths
        
        Args:
            frame: Input frame
            player_paths: Dictionary mapping player IDs to their path history
            player_teams: Dictionary mapping player IDs to team indices
            path_length: Maximum path length to display
            
        Returns:
            Frame with player paths
        """
        vis_frame = frame.copy()
        
        for player_id, path in player_paths.items():
            # Get team color
            team = player_teams.get(player_id) if player_teams else None
            color = self.team_colors.get(team, (255, 255, 255))
            
            # Limit path length
            if len(path) > path_length:
                path = path[-path_length:]
            
            # Draw path
            for i in range(1, len(path)):
                # Vary transparency based on recency
                alpha = 0.5 + 0.5 * (i / len(path))
                line_color = tuple([int(c * alpha) for c in color])
                cv2.line(vis_frame, 
                       (int(path[i-1][0]), int(path[i-1][1])), 
                       (int(path[i][0]), int(path[i][1])), 
                       line_color, 2)
        
        return vis_frame
    
    def draw_possession_indicator(self, frame, possession, team_names=None):
        """
        Draw ball possession indicator
        
        Args:
            frame: Input frame
            possession: Current possession (team_idx, player_id) or None
            team_names: Optional team names
            
        Returns:
            Frame with possession indicator
        """
        vis_frame = frame.copy()
        
        if possession is None:
            return vis_frame
            
        team_idx, player_id = possession
        
        # Get team name and color
        team_name = team_names[team_idx] if team_names and team_idx in team_names else f"Team {team_idx+1}"
        color = self.team_colors.get(team_idx, (255, 255, 255))
        
        # Draw semi-transparent rectangle
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
        
        # Draw text
        cv2.putText(vis_frame, f"Ball Possession:", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"{team_name} (ID: {player_id})", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_frame
    
    def draw_stats(self, frame, team_stats, team_names=None):
        """
        Draw team statistics
        
        Args:
            frame: Input frame
            team_stats: Dictionary with team statistics
            team_names: Optional team names
            
        Returns:
            Frame with statistics
        """
        vis_frame = frame.copy()
        
        # Create semi-transparent overlay for stats panel
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (frame.shape[1] - 320, 10), (frame.shape[1] - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        y_offset = 30
        for team_idx in [0, 1]:
            team_name = team_names[team_idx] if team_names and team_idx in team_names else f"Team {team_idx+1}"
            color = self.team_colors.get(team_idx, (255, 255, 255))
            
            # Get team stats
            possession = team_stats.get('possession', {}).get(team_idx, 0)
            avg_speed = team_stats.get('avg_speed', {}).get(team_idx, 0)
            distance = team_stats.get('total_distance', {}).get(team_idx, 0)
            
            # Draw team name and stats
            cv2.putText(vis_frame, f"{team_name}:", (frame.shape[1] - 310, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis_frame, f"Possession: {possession:.1f}%", (frame.shape[1] - 310, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_frame, f"Avg Speed: {avg_speed:.1f} m/s", (frame.shape[1] - 310, y_offset + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 60
        
        return vis_frame
