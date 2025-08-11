import time
import hydra
import argparse
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import torch
from dataloaders.semantic_kitty import SemanticKITTIDataset, kitti_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.temporal_point_transformer import TemporalPointTransformer
from omegaconf import OmegaConf
# import hdbscan
import matplotlib.pyplot as plt
from pathlib import Path

OmegaConf.register_new_resolver("repeat", lambda value, n: [value] * n)

import colorsys

def make_hsv_palette(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    return [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues]

def main(cfg):
    rr.init("base", spawn=False)
    rr.connect_grpc()

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world/lidar/pred"),
            rrb.Spatial3DView(origin="world/lidar/gt"),
            # rrb.Spatial3DView(origin="world/lidar/dbscan"),
        ),
        collapse_panels=True
    )
    rr.send_blueprint(blueprint)

    # Dataset
    test_dataset = SemanticKITTIDataset(
        cfg.data.root_dir,
        cfg.data.test_sequences,
        num_pointclouds=cfg.data.num_pointclouds,
        transform_pointclouds=cfg.data.transform_pointclouds,
        add_timestamp_feat=cfg.data.add_timestamp_feat,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=kitti_collate_fn,
    )

    # Load model
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model = TemporalPointTransformer.load_from_checkpoint(cfg.checkpoint_path, cfg=cfg)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    index = 0
    for batch in tqdm(test_loader, desc="Running Inference"):
        rr.set_time("frame", sequence=index)

        # Move batch to device
        batch_device = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Inference
        with torch.no_grad():
            preds = model.infer(batch_device)
            preds = (preds > 0.5).float().cpu().numpy()

        preds = preds.astype(np.uint8).flatten()
        gt_mask = batch['masks'].cpu().numpy().astype(np.uint8).flatten()
        
        # TP / FP / FN / TN
        tp = (preds == 1) & (gt_mask > 0)
        fp = (preds == 1) & (gt_mask == 0)
        fn = (preds == 0) & (gt_mask > 0)
        tn = (preds == 0) & (gt_mask == 0)

        pred_colors = np.zeros((preds.shape[0], 3), dtype=np.uint8)
        pred_colors[tp] = [0, 255, 0]     # Green
        pred_colors[fp] = [255, 0, 0]     # Red
        pred_colors[fn] = [0, 0, 255]     # Blue
        pred_colors[tn] = [128, 128, 128] # Gray

        gt_colors = np.zeros((preds.shape[0], 3), dtype=np.uint8)
        gt_colors[gt_mask > 0] = [0, 255, 255]    # Cyan
        gt_colors[gt_mask == 0] = [128, 128, 128] # Gray

        latest_mask = (batch["timestamps"] == 1)
        points_np = batch["points"][latest_mask].cpu().numpy()
        pred_colors = pred_colors[latest_mask.cpu().numpy()]
        gt_colors = gt_colors[latest_mask.cpu().numpy()]

        # Log to rerun
        rr.log("world/lidar/pred", rr.Points3D(positions=points_np[:, :3], colors=pred_colors, radii=0.05))
        rr.log("world/lidar/gt", rr.Points3D(positions=points_np[:, :3], colors=gt_colors, radii=0.05))
        #
        # clusterer = hdbscan.HDBSCAN(
        #     min_cluster_size=20,  # smallest size of clusters you care about
        #     min_samples=5,       # how conservative to be: larger => more points labeled noise
        #     metric='euclidean',   # distance metric
        #     cluster_selection_epsilon=0.0,  # like a “soft eps” to split clusters further
        # )
        #
        # # above_ground = points_np[:, 2] > -1.1
        # # points_np = points_np[above_ground]
        # cluster_ids = clusterer.fit_predict(points_np)
        # unique_ids = np.unique(cluster_ids)
        # n_clusters = len(unique_ids)
        #
        # # Use any big colormap you want
        # cmap = plt.get_cmap("nipy_spectral", n_clusters)
        #
        # palette = make_hsv_palette(n_clusters)
        # id_to_color = {
        #     cid: (0.0, 0.0, 0.0) if cid == -1 else palette[i]
        #     for i, cid in enumerate(unique_ids)
        # }
        #
        # # Step 2: Map each point's cluster ID to its RGB color
        # cluster_colors = np.array([id_to_color[cid] for cid in cluster_ids])
        #
        # # Step 3: Log it
        # rr.log("world/lidar/dbscan", rr.Points3D(
        #     positions=points_np[:, :3],
        #     colors=cluster_colors,
        #     radii=0.05,
        # ))

        index += 1

    # Hold Rerun viewer open
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        rr.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--config_path", type=Path, required=True, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to the .ckpt file")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config_path)
    cfg.checkpoint_path = args.checkpoint_path
    main(cfg)
