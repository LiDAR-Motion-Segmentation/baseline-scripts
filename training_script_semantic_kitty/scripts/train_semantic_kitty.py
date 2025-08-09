import os
import wandb
import hydra
import torch
import lightning as L
from torch.utils.data import DataLoader
from dataloaders.semantic_kitty import SemanticKITTIDataset, kitti_collate_fn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf

from models.temporal_point_transformer import TemporalPointTransformer
from dotenv import load_dotenv

OmegaConf.register_new_resolver("repeat", lambda value, n: [value] * n)

@hydra.main(config_path="/scratch/soumo_roy/temporal-point-transformer/config", config_name="semantic_kitty_config")
def run(cfg):
    torch.set_float32_matmul_precision('high')
    L.seed_everything(cfg.training.seed, workers=True)

    semantic_kitty_train = SemanticKITTIDataset(cfg.data.root_dir, 
                             cfg.data.train_sequences, 
                             cfg.data.num_pointclouds, 
                             transform_pointclouds=cfg.data.transform_pointclouds,
                             apply_augmentations=cfg.data.apply_augmentations,
                             add_timestamp_feat=cfg.data.add_timestamp_feat,
                             )
    semantic_kitty_val = SemanticKITTIDataset(cfg.data.root_dir, 
                           cfg.data.val_sequences, 
                           cfg.data.num_pointclouds, 
                           transform_pointclouds=cfg.data.transform_pointclouds,
                           add_timestamp_feat=cfg.data.add_timestamp_feat,
                           apply_augmentations=False,
                           )
    semantic_kitty_test = SemanticKITTIDataset(cfg.data.root_dir, 
                            cfg.data.test_sequences, 
                            cfg.data.num_pointclouds,
                            transform_pointclouds=cfg.data.transform_pointclouds,
                            add_timestamp_feat=cfg.data.add_timestamp_feat,
                            apply_augmentations=False,
                            )

    train_dataloader = DataLoader(
        semantic_kitty_train,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn = kitti_collate_fn
    )
    val_dataloader = DataLoader(
        semantic_kitty_val,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn = kitti_collate_fn
    )
    test_dataloader = DataLoader(
        semantic_kitty_test,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn = kitti_collate_fn
    )

    load_dotenv()
    api_key = os.getenv('WANDB_API_KEY')
    if api_key:
        wandb.login(key=api_key)
        print("Successfully logged in to WandB.")
    else:
        print("WandB API key not found. Please ensure your .env file is configured properly. Logging locally")
    
    wandb_logger = WandbLogger(
        save_dir=cfg.logging.wandb.run_root_dir,
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        log_model=cfg.logging.wandb.log_model,
        save_code=cfg.logging.wandb.save_code,
        group=cfg.logging.wandb.group,
        name=cfg.logging.wandb.name,
        resume=cfg.logging.wandb.resume
    )

    wandb_run_dir = wandb_logger.experiment.dir

    checkpoint_dir = os.path.join(wandb_logger.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    OmegaConf.save(cfg, os.path.join(wandb_logger.save_dir, "..", "config.yaml"))

    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    checkpoint_callback = ModelCheckpoint(
        monitor='val/iou',             
        mode='max',                     
        save_top_k=1,                   
        dirpath=checkpoint_dir,
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        verbose=True,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    print(f"wandb_dir: {wandb_run_dir}")
    print(f"checkpoint_dir: {checkpoint_dir}")

    model = TemporalPointTransformer(cfg)

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        log_every_n_steps=cfg.training.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback,
                   lr_monitor],
    )

    trainer.fit(model, train_dataloader, val_dataloader)

run()
