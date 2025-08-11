# baseline-scripts
Extra utility codes used to run baselines smoothly

## semantic kitty scripts
### the files are placed in training_script_semantic_kitty directory

- activate the evironment (the environment file is environment.yml)
```
conda env create -f environment.yml
conda activate lidar_moseg
```
- to run and test the train dataloader and test dataloader script (move this script to the dataloader directory in codebase which ever is used)
```
python3 dataloaders/semantic_kitty.py
```
- to run the training script use (-m to be used when running in a module otherwise it not reqired also put this code in scripts directory)
```
tmux new -s training
tmux a -t training
python3 -m scripts.train_semantic_kitty.py 
```
- to use the first GPU on your system incase  it is not detected use this
```
export CUDA_VISIBLE_DEVICES=0
```
- To run the evaluation script for semantic kitty use
```
python3 -m scripts.eval_semantic_kitty --config_path <path>/config/semantic_kitty_config.yaml --checkpoint_path <path>/best-checkpoint-epoch=07-val_loss=0.00-v1.ckpt
```

- the utilities for point cloud  processing is present in `pointcloud_utils.py` and the config used for training along with the split is present in config folder in `semantic_kitty_config.yaml`