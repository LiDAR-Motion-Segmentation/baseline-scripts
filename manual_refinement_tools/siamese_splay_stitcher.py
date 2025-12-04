import argparse
import json
import numpy as np
import cv2
import torch
import torchreid
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cosine


class Node:
    def __init__(self, key, data) -> None:
        self.key = key  # Key: Last Frame Seen
        self.data = data  # Data: Tracklet Object
        self.left = None
        self.right = None


class SplayTree:
    # Recently accessed 'Lost Tracks' move to the root
    def __init__(self) -> None:
        self.root = None

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        y.right = x
        return y

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        y.left = x
        return y

    def _splay(self, root, key):
        if root is None or root.key == key:
            return root

        # key lies in left subtree
        if key < root.key:
            if root.left is None:
                return root

            # Zig-Zig (Left Left)
            if key < root.left.key:
                root.left.left = self._splay(root.left.left, key)
                root = self._right_rotate(root)

            # Zig-Zag (Left Right)
            elif key > root.left.key:
                root.left.right = self._splay(root.left.left, key)
                if root.left.right:
                    root.left = self._right_rotate(root.left)

            return self._right_rotate(root) if root.left else root

        # key lies in rigth subtree
        else:
            if root.right is None:
                return root

            # Zag-Zag (Right Right)
            if key > root.right.key:
                root.right.right = self._splay(root.right.right, key)
                root = self._left_rotate(root)

            # Zag-zig (Right Left)
            elif key < root.right.key:
                root.right.left = self._splay(root.right.left, key)
                if root.right.left:
                    root.right = self._right_rotate(root.right)

            return self._left_rotate(root) if root.right else root

    def insert(self, key, data):
        if self.root is None:
            self.root = Node(key, data)
            return

        self.root = self._splay(self.root, key)
        if self.root.key == key:
            # Update existing (Shouldn't happen for unique tracks, but safe)
            self.root.data = data
            return

        new_node = Node(key, data)
        if key < self.root.key:
            new_node.right = self.root
            new_node.left = self.root.link
            self.root.left = None
        else:
            new_node.left = self.root
            new_node.right = self.root.right
            self.root.right = None
        self.root = new_node

    def find_clossest(self, target_key, max_gap=30):
        # Custom search: Find a node with key close to target_key.
        # Used to find tracks that ended 'recently'
        if self.root is None:
            return None

        # Splay around the target key to bring close nodes to top
        self.root = self._splay(self.root, target_key)

        # Check Root
        if abs(self.root.key - target_key) <= max_gap:
            return self.root.data

        # Check neighbors (simplified for code brevity)
        # Ideally we search the localized tree area
        return None


class SiameseEngine:
    def __init__(self, device="cuda") -> None:
        print("Loading Siamese Network (OSNet)")
        self.model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, loss="softmax", pretrained=True
        )
        self.model.to(device).eval()
        self.device = device

    def extract_features(self, img_crop):
        """Runs the 'Siamese' forward pass on one image branch"""
        if img_crop.size == 0:
            return None

        # Preprocess
        crop = cv2.resize(img_crop, (256, 128))
        tensor = (
            torch.from_numpy(crop).permute(2, 0, 1).float().div(255.0).to(self.device)
        )
        tensor = tensor.unsqueeze(0)

        # Norm
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            feat = self.model(tensor)
        return feat.cpu().numpy().flatten()

    def compute_similarity(self, feat1, feat2):
        """Returns Cosine Similarity (1.0 = Same, -1.0 = Opposite)"""
        if feat1 is None or feat2 is None:
            return 0.0
        return 1.0 - cosine(feat1, feat2)


class Tracklet:
    def __init__(self, track_id) -> None:
        self.id = track_id
        self.frames = []
        self.features = []  # list of feature vectors
        self.avg_features = None
        self.start_frame = float("inf")
        self.end_frame = float("-inf")

    def finalize(self):
        """Compute average embedding for the tracklet"""
        if self.features:
            self.avg_features = np.mean(self.features, axis=0)


def run_stitching(json_dir, img_dir, out_dir):
    json_dir, img_dir, out_dir = Path(json_dir), Path(img_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    siamese = SiameseEngine()
    lost_tracks_tree = SplayTree()  # The "Smart Cache"

    print(" Building Tracklets from JSONs")
    json_files = sorted(list(json_dir.glob("*.json")))

    # Mapping: {frame_idx: [list of detections]}
    all_detections = {}

    # global map to store tracklets: {track_id: Tracklet}
    active_tracklets = {}
    finished_tracklets = []

    for f_path in tqdm(json_files, desc="Extracting Features"):
        frame_idx = int(f_path.stem)

        img_path = img_dir / f"{f_path.stem}.jpg"
        if not img_path.exists():
            continue
        frame_img = cv2.imread(str(img_path))

        with open(f_path, "r") as f:
            data = json.load(f)

        current_frame_ids = set()

        for obj in data:
            tid = obj["global_id"]
            bbox = list(map(int, obj["bbox"]))
            current_frame_ids.add(tid)

            # init tracklet if new
            if tid not in active_tracklets:
                active_tracklets[tid] = Tracklet(tid)
                active_tracklets[tid].start_frame = frame_idx

            t = active_tracklets[tid]
            t.end_frame = frame_idx

            # Extract Feature (Siamese Branch 1)
            # Optimization: Only extract for every 5th frame to save time
            if frame_idx % 5 == 0:
                h, w, _ = frame_img.shape
                x1, y1, x2, y2 = bbox

                # clip
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame_img[y1:y2, x1:x2]

                feat = siamese.extract_features(crop)
                if feat is not None:
                    t.features.append(feat)

        # Check for tracks that ended
        # If a track hasn't been seen for 5 frames, move to "Finished"

    for t in active_tracklets.values():
        t.finalize()
        finished_tracklets.append(t)

    # Sort by Start Time for processing
    finished_tracklets.sort(key=lambda x: x.start_frame)
    print(f"Found {len(finished_tracklets)} fragments. Stitching")

    # Remapping table: {old_id: new_id}
    id_map = {}

    for current_track in tqdm(finished_tracklets, desc="Stitching"):
        # resolve id if already mapped
        if current_track.id in id_map:
            current_track.id = id_map[current_track.id]

        # Search Splay Tree for a "Lost Track" that ended just before this one starts
        # Search Key = Current Start Frame - 1
        search_key = current_track.start_frame

        # find candidate in slay tree, 60 frames gap allowed
        candidate = lost_tracks_tree.find_clossest(search_key, max_gap=60)
        matched = False
        if candidate:
            # SIAMESE MATCHING
            # Compare Avg Feature of Candidate vs Current
            sim = siamese.compute_similarity(
                candidate.avg_feature, current_track.avg_feature
            )

            # threshold for the same person
            if sim > 0.05:
                # merging here need to check
                # map current ID -> candidate ID
                id_map[current_track.id] = candidate.id
                current_track.id = candidate.id

                # update Candidate Info (Extend end frame)
                candidate.end_frame = current_track.end_frame

                # update features (weighted avg)
                if current_track.avg_feature is not None:
                    candidate.avg_feature = (
                        candidate.avg_feature + current_track.avg_feature
                    ) / 2.0

                # Re-insert into Splay Tree with NEW end time
                # (Splay tree moves this updated track to root)
                lost_tracks_tree.insert(candidate.end_frame, candidate)
                matched = True

        if not matched:
            # If no match, insert this new track into the tree as a potential candidate for future tracks
            lost_tracks_tree.insert(current_track.end_frame, current_track)

    print(" Saving Corrected JSON")
    for f_path in json_files:
        with open(f_path, "r") as f:
            data = json.load(f)

        changed = False
        for obj in data:
            gid = obj["global_id"]
            if gid in id_map:
                # Apply the recursive map (in case of chain merges 5->10->15)
                while gid in id_map:
                    gid = id_map[gid]
                obj["global_id"] = gid
                changed = True

        # Save
        if changed or True:  # Always save copy
            with open(out_dir / f_path.name, "w") as f:
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    run_stitching(args.json_dir, args.img_dir, args.out_dir)
