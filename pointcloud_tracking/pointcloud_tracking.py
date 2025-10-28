#!/usr/bin/env python3
"""
3D Point Cloud Human Tracking with Motion Analysis
Integrates M2Track/MMTrack with PointPillars for human detection and tracking
Supports moving vs static object classification and JSON export
"""

import os
import glob
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import math

# Core libraries for point cloud processing
try:
    import open3d as o3d
    import torch
    import torch.nn as nn
    from scipy.spatial.distance import cdist
    from sklearn.cluster import DBSCAN
except ImportError as e:
    print(f"Please install required dependencies: {e}")
    print("pip install open3d-python torch scikit-learn scipy")
    exit(1)

# Optional: Try to import tracking libraries
try:
    # These would be the actual M2Track/MMTrack implementations
    # from open3dsot.models import M2Track
    # from mmtrack.apis import inference_mot, init_model
    HAS_TRACKING_LIBS = False
    print("Warning: M2Track/MMTrack libraries not found. Using fallback tracking.")
except ImportError:
    HAS_TRACKING_LIBS = False

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox3D:
    """3D Bounding Box representation"""

    center: np.ndarray  # [x, y, z]
    size: np.ndarray  # [length, width, height]
    rotation: float  # rotation around z-axis in radians
    confidence: float = 1.0
    obj_id: str = ""
    obj_type: str = "unknown"


@dataclass
class TrackingResult:
    """Tracking result for a single frame"""

    frame_id: int
    timestamp: float
    boxes: List[BoundingBox3D]
    point_cloud_path: str


class MotionAnalyzer:
    """Analyzes motion patterns to classify objects as static or moving"""

    def __init__(self, history_length: int = 10, motion_threshold: float = 0.1):
        self.history_length = history_length
        self.motion_threshold = motion_threshold  # meters
        self.track_histories = defaultdict(lambda: deque(maxlen=history_length))

    def update_track(self, obj_id: str, position: np.ndarray, timestamp: float):
        """Update position history for a tracked object"""
        self.track_histories[obj_id].append(
            {"position": position.copy(), "timestamp": timestamp}
        )

    def classify_motion(self, obj_id: str) -> str:
        """Classify object as moving or static based on position history"""
        if obj_id not in self.track_histories:
            return "unknown"

        history = list(self.track_histories[obj_id])
        if len(history) < 3:
            return "unknown"

        # Calculate total displacement over time window
        positions = np.array([h["position"] for h in history])

        # Calculate velocity magnitudes between consecutive frames
        velocities = []
        for i in range(1, len(positions)):
            dt = history[i]["timestamp"] - history[i - 1]["timestamp"]
            if dt > 0:
                displacement = np.linalg.norm(positions[i] - positions[i - 1])
                velocity = displacement / dt
                velocities.append(velocity)

        if not velocities:
            return "people_static"

        avg_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)

        # Classification logic
        if (
            avg_velocity > self.motion_threshold
            and max_velocity > self.motion_threshold * 2
        ):
            return "moving_people"
        else:
            return "people_static"


class PointPillarsDetector:
    """PointPillars-based human detection (fallback implementation)"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.5

        # Human-specific parameters (typical human dimensions in meters)
        self.human_size_range = {
            "length": (0.3, 0.8),  # 30cm to 80cm
            "width": (0.3, 0.8),  # 30cm to 80cm
            "height": (1.2, 2.2),  # 1.2m to 2.2m
        }

        logger.info(f"Initialized PointPillars detector on {device}")

    def detect_humans(self, point_cloud: np.ndarray) -> List[BoundingBox3D]:
        """
        Detect humans in point cloud using PointPillars
        Fallback implementation using geometric clustering
        """
        if point_cloud.shape[0] < 100:  # Too few points
            return []

        # Filter points to human height range (0.5m to 2.5m above ground)
        ground_z = np.percentile(point_cloud[:, 2], 5)  # Estimate ground level
        human_height_mask = (point_cloud[:, 2] > ground_z + 0.5) & (
            point_cloud[:, 2] < ground_z + 2.5
        )
        filtered_points = point_cloud[human_height_mask]

        if len(filtered_points) < 50:
            return []

        # Cluster points using DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=20).fit(filtered_points[:, :3])
        labels = clustering.labels_

        detections = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in unique_labels:
            cluster_points = filtered_points[labels == label]

            if len(cluster_points) < 20:
                continue

            # Compute bounding box
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)

            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords

            # Filter by human-like dimensions
            if (
                self.human_size_range["length"][0]
                <= size[0]
                <= self.human_size_range["length"][1]
                and self.human_size_range["width"][0]
                <= size[1]
                <= self.human_size_range["width"][1]
                and self.human_size_range["height"][0]
                <= size[2]
                <= self.human_size_range["height"][1]
            ):

                # Estimate orientation (simplified)
                rotation = 0.0  # Could implement PCA-based orientation estimation

                bbox = BoundingBox3D(
                    center=center[:3],
                    size=size[:3],
                    rotation=rotation,
                    confidence=min(1.0, len(cluster_points) / 100.0),
                    obj_type="people_unknown",
                )
                detections.append(bbox)

        logger.info(f"Detected {len(detections)} human candidates")
        return detections


class SimpleTracker:
    """Simple 3D object tracker (fallback for M2Track)"""

    def __init__(self, max_distance: float = 2.0, max_disappeared: int = 5):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_id = 1
        self.active_tracks = {}
        self.disappeared = {}

    def update(self, detections: List[BoundingBox3D]) -> List[BoundingBox3D]:
        """Update tracker with new detections"""
        if not detections:
            # Mark all existing tracks as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.active_tracks[track_id]
                    del self.disappeared[track_id]
            return []

        # If no existing tracks, initialize with all detections
        if not self.active_tracks:
            for detection in detections:
                track_id = str(self.next_id)
                detection.obj_id = track_id
                self.active_tracks[track_id] = detection
                self.next_id += 1
            return detections

        # Compute distance matrix between existing tracks and detections
        track_positions = np.array(
            [track.center for track in self.active_tracks.values()]
        )
        detection_positions = np.array([det.center for det in detections])

        distance_matrix = cdist(track_positions, detection_positions)

        # Hungarian assignment (simplified greedy assignment)
        used_detection_indices = set()
        updated_tracks = []

        for i, (track_id, track) in enumerate(self.active_tracks.items()):
            if i >= len(distance_matrix):
                continue

            min_distance_idx = np.argmin(distance_matrix[i])
            min_distance = distance_matrix[i, min_distance_idx]

            if (
                min_distance < self.max_distance
                and min_distance_idx not in used_detection_indices
            ):
                # Update existing track
                detections[min_distance_idx].obj_id = track_id
                self.active_tracks[track_id] = detections[min_distance_idx]
                updated_tracks.append(detections[min_distance_idx])
                used_detection_indices.add(min_distance_idx)

                # Remove from disappeared
                if track_id in self.disappeared:
                    del self.disappeared[track_id]
            else:
                # Track disappeared
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                if self.disappeared[track_id] <= self.max_disappeared:
                    updated_tracks.append(track)

        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detection_indices:
                track_id = str(self.next_id)
                detection.obj_id = track_id
                self.active_tracks[track_id] = detection
                updated_tracks.append(detection)
                self.next_id += 1

        # Remove tracks that have been disappeared for too long
        for track_id in list(self.disappeared.keys()):
            if self.disappeared[track_id] > self.max_disappeared:
                del self.active_tracks[track_id]
                del self.disappeared[track_id]

        return updated_tracks


class PointCloudTrackingSystem:
    """Main tracking system coordinating detection, tracking, and motion analysis"""

    def __init__(self, output_dir: str, model_path: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.detector = PointPillarsDetector(model_path)
        self.tracker = SimpleTracker()
        self.motion_analyzer = MotionAnalyzer()

        # Results storage
        self.tracking_results = []

    def load_point_cloud(self, pcd_path: str) -> np.ndarray:
        """Load point cloud from PCD file"""
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)

            # If point cloud has colors or other attributes, include them
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                points = np.hstack([points, colors])

            logger.info(f"Loaded point cloud with {len(points)} points from {pcd_path}")
            return points

        except Exception as e:
            logger.error(f"Error loading point cloud {pcd_path}: {e}")
            return np.array([])

    def process_frame(self, frame_id: int, pcd_path: str) -> TrackingResult:
        """Process a single frame"""
        timestamp = time.time()

        # Load point cloud
        point_cloud = self.load_point_cloud(pcd_path)
        if len(point_cloud) == 0:
            return TrackingResult(frame_id, timestamp, [], pcd_path)

        # Detect humans
        detections = self.detector.detect_humans(point_cloud)

        # Track detections
        tracked_objects = self.tracker.update(detections)

        # Update motion analysis and classify motion
        for obj in tracked_objects:
            self.motion_analyzer.update_track(obj.obj_id, obj.center, timestamp)
            obj.obj_type = self.motion_analyzer.classify_motion(obj.obj_id)

        result = TrackingResult(frame_id, timestamp, tracked_objects, pcd_path)
        self.tracking_results.append(result)

        logger.info(f"Frame {frame_id}: {len(tracked_objects)} tracked objects")
        return result

    def export_frame_json(self, result: TrackingResult) -> str:
        """Export single frame results to JSON format"""
        json_data = []

        for box in result.boxes:
            obj_data = {
                "obj_id": box.obj_id,
                "obj_type": box.obj_type,
                "psr": {
                    "position": {
                        "x": float(box.center[0]),
                        "y": float(box.center[1]),
                        "z": float(box.center[2]),
                    },
                    "rotation": {"x": 0.0, "y": 0.0, "z": float(box.rotation)},
                    "scale": {
                        "x": float(box.size[0]),
                        "y": float(box.size[1]),
                        "z": float(box.size[2]),
                    },
                },
            }
            json_data.append(obj_data)

        # Save to file
        output_path = self.output_dir / f"frame_{result.frame_id:06d}.json"
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Exported frame {result.frame_id} to {output_path}")
        return str(output_path)

    def process_directory(self, pcd_directory: str):
        """Process all PCD files in directory"""
        pcd_dir = Path(pcd_directory)

        # Find all PCD files and sort numerically
        pcd_files = list(pcd_dir.glob("*.pcd"))

        # Sort by numeric value in filename
        def extract_number(filename):
            import re

            numbers = re.findall(r"\d+", filename.stem)
            return int(numbers[0]) if numbers else 0

        pcd_files.sort(key=extract_number)

        if not pcd_files:
            logger.error(f"No PCD files found in {pcd_directory}")
            return

        logger.info(f"Found {len(pcd_files)} PCD files to process")

        # Process each frame
        for frame_id, pcd_path in enumerate(pcd_files):
            logger.info(
                f"Processing frame {frame_id + 1}/{len(pcd_files)}: {pcd_path.name}"
            )

            result = self.process_frame(frame_id, str(pcd_path))

            # Export JSON for this frame
            self.export_frame_json(result)

        # Export summary statistics
        self.export_summary()

    def export_summary(self):
        """Export tracking summary and statistics"""
        summary = {
            "total_frames": len(self.tracking_results),
            "total_unique_tracks": len(
                set(
                    obj.obj_id
                    for result in self.tracking_results
                    for obj in result.boxes
                )
            ),
            "motion_classification": {},
            "detection_statistics": {
                "frames_with_detections": sum(
                    1 for r in self.tracking_results if r.boxes
                ),
                "total_detections": sum(len(r.boxes) for r in self.tracking_results),
                "avg_detections_per_frame": sum(
                    len(r.boxes) for r in self.tracking_results
                )
                / max(1, len(self.tracking_results)),
            },
        }

        # Count motion classifications
        motion_counts = defaultdict(int)
        for result in self.tracking_results:
            for obj in result.boxes:
                motion_counts[obj.obj_type] += 1
        summary["motion_classification"] = dict(motion_counts)

        # Save summary
        summary_path = self.output_dir / "tracking_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported tracking summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="3D Point Cloud Human Tracking with Motion Analysis"
    )
    parser.add_argument("pcd_directory", help="Directory containing PCD files")
    parser.add_argument(
        "--output",
        "-o",
        default="./tracking_output",
        help="Output directory for JSON files",
    )
    parser.add_argument("--model", "-m", help="Path to PointPillars model checkpoint")
    parser.add_argument(
        "--motion-threshold",
        "-t",
        type=float,
        default=0.1,
        help="Motion threshold in meters/second (default: 0.1)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input directory
    if not os.path.exists(args.pcd_directory):
        logger.error(f"Input directory does not exist: {args.pcd_directory}")
        return

    # Initialize tracking system
    tracking_system = PointCloudTrackingSystem(
        output_dir=args.output, model_path=args.model
    )

    # Set motion threshold
    tracking_system.motion_analyzer.motion_threshold = args.motion_threshold

    # Process directory
    logger.info(f"Starting processing of {args.pcd_directory}")
    start_time = time.time()

    try:
        tracking_system.process_directory(args.pcd_directory)

        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
