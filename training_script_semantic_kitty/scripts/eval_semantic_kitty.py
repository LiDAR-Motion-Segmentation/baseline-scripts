import os
import torch
import hydra
import argparse
from omegaconf import OmegaConf
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path

import lightning as L
from models.temporal_point_transformer import TemporalPointTransformer
from dataloaders.semantic_kitty import SemanticKITTIDataset, kitti_collate_fn
from hydra.experimental import initialize, compose

OmegaConf.register_new_resolver("repeat", lambda value, n: [value] * n)

def run(cfg):
    L.seed_everything(cfg.training.seed, workers=True)

    test_sequences = cfg.data.test_sequences

    overall_metrics = defaultdict(list)

    for test_sequence in test_sequences:
        print(f"Processing sequence: {test_sequence}")

        test_dataset = SemanticKITTIDataset(
            cfg.data.root_dir, 
            [test_sequence],  
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

        print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
        model = TemporalPointTransformer.load_from_checkpoint(cfg.checkpoint_path, cfg=cfg)

        trainer = L.Trainer(
            accelerator=cfg.training.accelerator,
            devices=cfg.training.devices,
            logger=False,
            enable_checkpointing=False,
        )

        print(f"Running test for sequence {test_sequence}...")
        trainer.test(model, dataloaders=test_loader, verbose=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--config_path", type=Path, required=True, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to the .ckpt file")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config_path)
    cfg.checkpoint_path = args.checkpoint_path
    run(cfg)
