# running the model on semantic kitty dataset
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
from utils.pointcloud_utils import canonical_transform, random_augment, crop_square_region
from torch.utils.data import DataLoader

def kitti_collate_fn(batch):
    sequences = []
    all_points = []
    all_coords = []
    all_masks = []
    all_timestamps = []
    all_batch_indices = []
    all_offsets = []
    all_poses = []

    cum_offset = 0
    batch_id = 0

    for sample in batch:
        sequence, pointclouds, timestamps_list, offsets, masks, poses = sample
        sequences.append(sequence)

        for n in range(len(pointclouds)):
            pc = torch.from_numpy(pointclouds[n]).float()        # (N_i,4)
            mask = torch.from_numpy(masks[n]).float()            # (N_i,)
            mask = torch.where(mask > 0, 1.0, 0.0)
            timestamps = torch.from_numpy(timestamps_list[n]).float().squeeze(1)  # (N_i,)

            all_points.append(pc)
            all_coords.append(pc[:, :3])
            all_masks.append(mask)
            all_timestamps.append(timestamps)
            all_batch_indices.append(torch.full((pc.shape[0],), batch_id, dtype=torch.long))

            all_offsets.append(cum_offset + pc.shape[0])
            cum_offset += pc.shape[0]

        all_poses.append(torch.tensor(poses, dtype=torch.float32))
        batch_id += 1

    return {
        "sequences": sequences,
        "points": torch.cat(all_points, dim=0),
        "coords": torch.cat(all_coords, dim=0),
        "masks": torch.cat(all_masks, dim=0),
        "timestamps": torch.cat(all_timestamps, dim=0),
        "batch_indices": torch.cat(all_batch_indices, dim=0),
        "offsets": torch.tensor(all_offsets, dtype=torch.long),
    }
    
def downsample_pointcloud(pc, mask, max_points=20000):
    """Downsample point cloud and corresponding mask to reduce memory usage"""
    if len(pc) > max_points:
        indices = np.random.choice(len(pc), max_points, replace=False)
        return pc[indices], mask[indices]
    return pc, mask

class SemanticKITTIDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 sequences: list,
                 num_pointclouds: int,
                 transform_pointclouds: bool = True,
                 apply_augmentations: bool = False,
                 add_timestamp_feat: bool = True):
        self.root_dir = root_dir
        self.sequences = sequences
        self.num_pointclouds = num_pointclouds
        self.transform_pointclouds = transform_pointclouds
        self.apply_augmentations = apply_augmentations
        self.add_timestamp_feat = add_timestamp_feat

        self.samples = []
        for seq in sequences:
            seq_dir = os.path.join(root_dir, "sequences", f"{int(seq):02d}")
            velodyne_dir = os.path.join(seq_dir, "velodyne")
            label_dir = os.path.join(seq_dir, "labels")
            times_file = os.path.join(seq_dir, "times.txt")
            poses_file = os.path.join(seq_dir, "poses.txt")

            # collect and sort
            pc_files = sorted(os.listdir(velodyne_dir))
            mask_files = sorted(os.listdir(label_dir))
            # load times
            times = np.loadtxt(times_file).reshape(-1, 1)
            # load poses
            poses = np.loadtxt(poses_file)[:, 1:8]  # skip timestamp

            assert len(pc_files) == len(mask_files) == len(poses), (
                f"Mismatch in number of samples:\n"
                f"- Point cloud files: {len(pc_files)}\n"
                f"- Mask files: {len(mask_files)}\n"
                f"- Poses: {len(poses)}\n"
                f"Ensure the dataset has all files"
            )

            num_frames = len(pc_files)
            for i in range(num_frames - self.num_pointclouds + 1):
                pc_window = [os.path.join(velodyne_dir, pc_files[j]) for j in range(i, i+self.num_pointclouds)]
                mask_window = [os.path.join(label_dir, mask_files[j]) for j in range(i, i+self.num_pointclouds)]
                pose_window = poses[i:i+num_pointclouds]
                self.samples.append((seq, pc_window, mask_window, pose_window))

        print(f"Loaded {len(self.samples)} samples from KITTI sequences {sequences}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, pc_files, mask_files, poses = self.samples[idx]

        pointclouds = []
        timestamps = []
        offsets = [0,]
        masks = []

        assert len(pc_files) == len(mask_files), (
            f"Mismatch in number of pointclouds ({len(pc_files)}) and masks ({len(mask_files)})"
        )
        
        for i in range(self.num_pointclouds):
            pc_path = pc_files[i]
            mask_path = mask_files[i]

            # load (why reshape here need to check)
            pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
            mask = np.fromfile(mask_path, dtype=np.uint32).reshape(-1)  # per-point label
            mask = (mask > 250).astype(np.float32)  # moving vs static

            pc, mask = crop_square_region(pc, mask, 10)
            pc, mask= downsample_pointcloud(pc, mask)
            # t = np.ones((pc.shape[0], 1), dtype=np.float32) * times[i]
            t = np.ones([pc.shape[0], 1])*i/(self.num_pointclouds-1)

            pointclouds.append(pc)
            masks.append(mask)
            timestamps.append(t)
            offsets.append(offsets[-1] + pc.shape[0])

        if self.transform_pointclouds:
            pointclouds = canonical_transform(pointclouds, poses)
        if self.apply_augmentations:
            pointclouds = random_augment(pointclouds)
        if self.add_timestamp_feat:
            for i in range(len(pointclouds)):
                pointclouds[i] = np.concatenate((pointclouds[i], timestamps[i]), axis=1)

        return sequence, pointclouds, timestamps, offsets[:self.num_pointclouds], masks, poses


# Example usage:
if __name__ == "__main__":
    root_dir = "/scratch/soumo_roy/semantic-kitty-dataset/dataset"
    train_sequences = list(range(0, 11))  # use 0–10 sequences
    num_pointclouds = 8
    test_sequences = [8]

    kitty_dataset_train = SemanticKITTIDataset(root_dir, train_sequences, num_pointclouds,
                                  transform_pointclouds=True,
                                  apply_augmentations=True,
                                  add_timestamp_feat=True)
    
    kitty_dataset_test = SemanticKITTIDataset(root_dir, test_sequences, num_pointclouds,
                                  transform_pointclouds=True,
                                  apply_augmentations=True,
                                  add_timestamp_feat=True) 

    train_loader = DataLoader(
        kitty_dataset_train,
        batch_size=2,
        shuffle=True,
        collate_fn=kitti_collate_fn,
        num_workers=4)
    
    test_loader = DataLoader(
        kitty_dataset_test,
        batch_size=1,
        shuffle=True,
        collate_fn=kitti_collate_fn,
        num_workers=4
    )
    
    for sample in tqdm(kitty_dataset_test):
        sequence, pointclouds, timestamps, offsets, masks, poses = sample

