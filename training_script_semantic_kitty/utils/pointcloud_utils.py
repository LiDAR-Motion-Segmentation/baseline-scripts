import numpy as np
from scipy.spatial.transform import Rotation as R

def crop_square_region(pc: np.ndarray, mask: np.ndarray, limit: float = 5.0):
    """Keeps only points where abs(x) < limit and abs(y) < limit."""
    keep = (np.abs(pc[:, 0]) < limit) & (np.abs(pc[:, 1]) < limit)
    return pc[keep], mask[keep]

def pad_pointcloud(point_cloud, mask, target_size=10000):
    current_size = point_cloud.shape[0]  # N, the current number of points
    feature_size = point_cloud.shape[1]  # C, the number of features per point
    
    # If the current size is less than the target, pad the point cloud and mask
    if current_size < target_size:
        padding_point_cloud = np.zeros((target_size - current_size, feature_size), dtype=point_cloud.dtype)
        padding_mask = np.full((target_size - current_size,), -1, dtype=mask.dtype)  # Padding label as -1

        padded_point_cloud = np.vstack([point_cloud, padding_point_cloud])
        padded_mask = np.concatenate([mask, padding_mask])
    else:
        # If the current size is larger than or equal to the target size, truncate the point cloud and mask
        padded_point_cloud = point_cloud[:target_size]
        padded_mask = mask[:target_size]

    return padded_point_cloud, padded_mask

def canonical_transform(pointclouds, poses):
    assert len(pointclouds) == len(poses), "Number of Pointclouds and Poses do not match"
    
    def pose_to_matrix(pose):
        pose_array = np.array(pose)
        # print(f"Debug: pose shape: {pose_array.shape}, content: {pose_array}")
        
        if len(pose_array) != 7:
            raise ValueError(f"Expected pose to have 7 elements, got {len(pose_array)}")
            
        t = pose_array[:3]
        q = pose_array[3:]
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = t
        return T

    transforms = [pose_to_matrix(p) for p in poses]
    T_ref_inv = np.linalg.inv(transforms[-1])
    # print(f"Debug: T_ref_inv shape: {T_ref_inv.shape}")

    transformed_pointclouds = []
    for i, (pc, T) in enumerate(zip(pointclouds, transforms)):
        pc = np.array(pc)
        # print(f"Debug: Point cloud {i} original shape: {pc.shape}")
        
        # Extract only x, y, z coordinates (first 3 columns)
        if pc.shape[1] < 3:
            raise ValueError(f"Point cloud {i} has only {pc.shape[1]} columns, need at least 3")
            
        pc_xyz = pc[:, :3]
        # print(f"Debug: Point cloud {i} xyz shape: {pc_xyz.shape}")
        
        pc_hom = np.hstack([pc_xyz, np.ones((pc_xyz.shape[0], 1))])
        # print(f"Debug: pc_hom shape: {pc_hom.shape}")
        # print(f"Debug: T shape: {T.shape}")
        # print(f"Debug: About to compute: T_ref_inv @ T @ pc_hom.T")
        # print(f"Debug: T_ref_inv.shape: {T_ref_inv.shape}, T.shape: {T.shape}, pc_hom.T.shape: {pc_hom.T.shape}")
        
        # Break down the matrix multiplication
        temp1 = T_ref_inv @ T
        # print(f"Debug: (T_ref_inv @ T) shape: {temp1.shape}")
        
        temp2 = pc_hom.T
        # print(f"Debug: pc_hom.T shape: {temp2.shape}")
        
        try:
            pc_transformed = (temp1 @ temp2).T[:, :3]
            transformed_pointclouds.append(pc_transformed)
            # print(f"Debug: Successfully transformed point cloud {i}")
        except Exception as e:
            print(f"Debug: Error in matrix multiplication: {e}")
            print(f"Debug: temp1 shape: {temp1.shape}, temp2 shape: {temp2.shape}")
            raise
    
    return transformed_pointclouds

def random_augment(pointclouds: list):
    '''Augment a set of pointclouds'''
    theta = np.random.uniform(0, 2 * np.pi)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    
    scale = np.random.uniform(0.8, 1.25)
    
    flip = np.random.rand() > 0.5
    
    augmented_pointclouds = []
    for pc in pointclouds:
        pc_aug = pc @ R.T
        pc_aug *= scale
        if flip:
            pc_aug[:, 0] = -pc_aug[:, 0]
        jitter = np.clip(0.01 * np.random.randn(*pc_aug.shape), -0.05, 0.05)
        pc_aug += jitter
        
        augmented_pointclouds.append(pc_aug)
    
    return augmented_pointclouds
