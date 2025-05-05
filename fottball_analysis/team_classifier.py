import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self, n_clusters=3):
        """
        Initialize team classifier using K-means clustering
        
        Args:
            n_clusters (int): Number of clusters for K-means (typically 3: team1, team2, other)
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.team_colors = None  # Will store representative colors for each team
        self.initialized = False
        
    def extract_player_color(self, frame, box, padding=5):
        """
        Extract dominant colors from player bounding box
        
        Args:
            frame: Input frame
            box: Player bounding box [x1, y1, x2, y2]
            padding: Pixels to remove from edges to avoid background
            
        Returns:
            Array of pixel colors from the player region
        """
        x1, y1, x2, y2 = box
        
        # Add padding to avoid edges (which might be background)
        x1 = max(0, x1 + padding)
        y1 = max(0, y1 + padding)
        x2 = min(frame.shape[1], x2 - padding)
        y2 = min(frame.shape[0], y2 - padding)
        
        # Extract the player region
        player_region = frame[y1:y2, x1:x2]
        
        if player_region.size == 0:
            return None
            
        # Reshape to list of pixels
        pixels = player_region.reshape(-1, 3)
        
        # Filter out grass/field colors (assuming green dominance)
        # This is a simple heuristic and might need adjustment
        non_field_mask = (pixels[:, 1] < pixels[:, 0] * 1.5) | (pixels[:, 1] < pixels[:, 2] * 1.5)
        filtered_pixels = pixels[non_field_mask]
        
        if len(filtered_pixels) == 0:
            return pixels  # Return original if filtering removed everything
            
        return filtered_pixels
        
    def cluster_team_colors(self, frame, player_boxes, min_players=6):
        """
        Cluster player colors to identify teams
        
        Args:
            frame: Input frame
            player_boxes: List of player bounding boxes
            min_players: Minimum number of players needed for reliable clustering
            
        Returns:
            Boolean indicating if clustering was successful
        """
        if len(player_boxes) < min_players:
            return False
            
        # Extract colors from all players
        all_colors = []
        
        for box in player_boxes:
            colors = self.extract_player_color(frame, box)
            if colors is not None and len(colors) > 0:
                # Take a sample of colors to avoid bias from large players
                sample_size = min(100, len(colors))
                indices = np.random.choice(len(colors), sample_size, replace=False)
                all_colors.append(colors[indices])
                
        if not all_colors:
            return False
            
        # Combine all samples
        combined_colors = np.vstack(all_colors)
        
        # Run K-means clustering
        self.kmeans.fit(combined_colors)
        cluster_centers = self.kmeans.cluster_centers_
        
        # Find the two clusters that are likely to be the teams
        # This assumes the third cluster is background, referee, etc.
        # Sort by distance from green (assumed field color)
        field_color = np.array([0, 255, 0])  # Green
        distances = np.linalg.norm(cluster_centers - field_color, axis=1)
        
        # The two furthest from green are likely team colors
        team_indices = np.argsort(distances)[-2:]
        self.team_colors = cluster_centers[team_indices]
        
        self.initialized = True
        return True
        
    def classify_player(self, frame, box):
        """
        Classify which team a player belongs to
        
        Args:
            frame: Input frame
            box: Player bounding box
            
        Returns:
            Team index (0 or 1) or None if cannot classify
        """
        if not self.initialized:
            return None
            
        player_colors = self.extract_player_color(frame, box)
        if player_colors is None or len(player_colors) == 0:
            return None
            
        # Sample from player colors
        sample_size = min(50, len(player_colors))
        indices = np.random.choice(len(player_colors), sample_size, replace=False)
        samples = player_colors[indices]
        
        # Predict cluster for each sampled pixel
        predictions = self.kmeans.predict(samples)
        
        # Count occurrences of each team's cluster
        team_0_count = np.sum(predictions == np.argmin(np.linalg.norm(self.kmeans.cluster_centers_ - self.team_colors[0], axis=1)))
        team_1_count = np.sum(predictions == np.argmin(np.linalg.norm(self.kmeans.cluster_centers_ - self.team_colors[1], axis=1)))
        
        # Assign to the majority team
        if max(team_0_count, team_1_count) < sample_size * 0.3:  # Not enough confidence
            return None
            
        return 0 if team_0_count > team_1_count else 1