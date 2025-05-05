import cv2
import numpy as np
import argparse
import os
import time
import pickle
from collections import defaultdict
import ultralytics

# Import project modules
from detector import ObjectDetector
from tracker import PlayerTracker, BallTracker
from team_classifier import TeamClassifier
from camera_motion import CameraMotionEstimator, PerspectiveTransformer
from analysis import SpeedDistanceCalculator, BallPossessionAnalyzer
from visualization import Visualizer
import utils

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Football Analysis System')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default=None, help='Path to output video file')
    parser.add_argument('--model', type=str, default='fottball_analysis/models/best.pt', help='Path to YOLO model weights')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--camera_comp', action='store_true', help='Enable camera motion compensation')
    parser.add_argument('--save_data', action='store_true', help='Save analysis data to file')
    parser.add_argument('--skip_frames', type=int, default=0, help='Number of frames to skip')
    parser.add_argument('--team1', type=str, default='Team 1', help='Name of team 1')
    parser.add_argument('--team2', type=str, default='Team 2', help='Name of team 2')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    return parser.parse_args()

def main():
    """Main function for football analysis system"""
    
    print("Script started.")  # Debug print to check if the script starts

    # Parse arguments
    args = parse_arguments()

    # Debug: Print video and model paths
    print(f"Video file path: {args.input}")
    print(f"Model path: {args.model}")

    # Check if input video exists
    if not os.path.isfile(args.input):
        print(f"Error: Input video file '{args.input}' does not exist.")
        return
    print(f"Absolute path of input: {os.path.abspath(args.input)}")

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.input}'.")
        return
    print("Video successfully opened.")

    # Debug: Check video info
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

    # Create output video writer
    video_writer, output_path = utils.create_output_video_writer(args.input, args.output)
    if video_writer is None:
        print("Error: Could not create output video writer.")
        return
    print(f"Output video will be saved to: {output_path}")

    # Initialize components
    detector = ObjectDetector(model_path=args.model, conf_threshold=args.conf)
    player_tracker = PlayerTracker(max_disappeared=30, max_distance=50)
    ball_tracker = BallTracker(max_disappeared=15, max_distance=30)
    team_classifier = TeamClassifier(n_clusters=3)
    camera_motion = CameraMotionEstimator() if args.camera_comp else None
    perspective_transform = PerspectiveTransformer()  # Not calibrated by default
    speed_calculator = SpeedDistanceCalculator(fps=fps)
    possession_analyzer = BallPossessionAnalyzer()
    visualizer = Visualizer()

    # Stats for analysis
    player_team_assignments = {}
    team_stats = {
        'possession': {0: 0.0, 1: 0.0},
        'avg_speed': {0: 0.0, 1: 0.0},
        'total_distance': {0: 0.0, 1: 0.0}
    }

    team_names = {0: args.team1, 1: args.team2}

    # Analysis data storage
    analysis_data = {
        'frame_data': [],
        'team_stats': defaultdict(list),
        'player_data': defaultdict(list),
        'possession_data': []
    }

    # Process video frames
    frame_count = 0
    team_classification_initialized = False

    print("Starting analysis...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read.")  # Debug if the video reaches the end
            break
        print(f"Processing frame {frame_count + 1}/{total_frames}")

        # Skip frames if needed
        if args.skip_frames > 0:
            if frame_count % (args.skip_frames + 1) != 0:
                frame_count += 1
                continue

        # Detect objects
        detections = detector.detect(frame)

        # Apply camera motion compensation if enabled
        if camera_motion is not None:
            camera_motion.estimate_motion(frame)

        # Track players
        player_centroids = []
        player_boxes = []

        for player in detections['players']:
            box = player['box']
            player_boxes.append(box)
            center = detector.get_center(box)
            player_centroids.append(center)

        player_objects = player_tracker.update(player_centroids)

        # Track ball
        ball_centroid = []
        if detections['ball'] is not None:
            ball_box = detections['ball']['box']
            ball_centroid = [detector.get_center(ball_box)]

        ball_objects = ball_tracker.update(ball_centroid)

        # Try to get interpolated ball position if not detected
        ball_position = None
        if len(ball_objects) > 0:
            ball_position = list(ball_objects.values())[0]
        else:
            ball_position = ball_tracker.get_interpolated_position()

        # Team classification with K-means clustering
        if not team_classification_initialized and len(player_boxes) >= 6:
            print("Initializing team classification...")
            if team_classifier.cluster_team_colors(frame, player_boxes):
                team_classification_initialized = True
                print("Team classification initialized!")

        # Assign players to teams
        for player_idx, (player_id, centroid) in enumerate(player_objects.items()):
            # Find the corresponding detection box
            if player_idx < len(player_boxes):
                box = player_boxes[player_idx]

                # Try to classify player if team classification is initialized
                if team_classification_initialized:
                    team = team_classifier.classify_player(frame, box)
                    player_tracker.update_team(player_id, team)
                    if team is not None:
                        player_team_assignments[player_id] = team

        # Calculate adjusted positions (camera motion compensation)
        adjusted_positions = {}
        for player_id, position in player_objects.items():
            if camera_motion is not None:
                adjusted_pos = camera_motion.adjust_point(position)
            else:
                adjusted_pos = position
            adjusted_positions[player_id] = adjusted_pos

        # Calculate speeds and distances
        player_speeds, player_total_distances = speed_calculator.update(adjusted_positions)

        # Ball possession analysis
        current_possession = possession_analyzer.update(
            ball_position,
            adjusted_positions,
            player_team_assignments,
            frame_count
        )

        # Update team statistics
        team_possession = possession_analyzer.get_possession_percentage()
        team_stats['possession'] = team_possession

        # Calculate average speed per team
        team_speeds = {0: [], 1: []}
        for player_id, speed in player_speeds.items():
            if player_id in player_team_assignments:
                team = player_team_assignments[player_id]
                team_speeds[team].append(speed)

        team_stats['avg_speed'] = {
            team: np.mean(speeds) if speeds else 0.0
            for team, speeds in team_speeds.items()
        }

        # Store analysis data
        if args.save_data:
            frame_data = {
                'frame_idx': frame_count,
                'detections': {
                    'players': len(detections['players']),
                    'ball': detections['ball'] is not None
                },
                'player_positions': adjusted_positions,
                'ball_position': ball_position,
                'possession': current_possession,
            }
            analysis_data['frame_data'].append(frame_data)

            for team in [0, 1]:
                analysis_data['team_stats'][team].append({
                    'frame_idx': frame_count,
                    'possession': team_possession.get(team, 0.0),
                    'avg_speed': team_stats['avg_speed'].get(team, 0.0)
                })

            if current_possession is not None:
                team_idx, player_id = current_possession
                analysis_data['possession_data'].append({
                    'frame_idx': frame_count,
                    'team': team_idx,
                    'player_id': player_id
                })

        # Visualization
        vis_frame = frame.copy()

        # Draw player paths
        player_paths = {}
        for player_id in player_objects.keys():
            player_paths[player_id] = speed_calculator.get_player_path(player_id, max_length=30)

        vis_frame = visualizer.draw_player_paths(vis_frame, player_paths, player_team_assignments)

        # Draw elliptical bounding boxes (update)
        vis_frame = visualizer.draw_elliptical_bounding_boxes(
            vis_frame,
            detections,
            player_team_assignments
        )

        # Draw possession indicator
        vis_frame = visualizer.draw_possession_indicator(
            vis_frame,
            current_possession,
            team_names
        )

        # Draw statistics
        vis_frame = visualizer.draw_stats(
            vis_frame,
            team_stats)

        # Write frame to output video
        video_writer.write(vis_frame)

        # Debug visualization (optional)
        if args.debug:
            print("Debug mode enabled")
            cv2.imshow('Football Analysis', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
    cap.release()
    video_writer.release()


if __name__ == '__main__':
    main()
