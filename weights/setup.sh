# YOLOv8 Model (you can choose others like yolov8x.pt for higher accuracy)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt

# Grounding-DINO Model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# SAM Model (base version)
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth

# Core AI and CV libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy

# YOLO and SAHI
pip install ultralytics sahi

# Grounding-DINO and SAM
pip install segment-anything-hq supervision
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# For downloading models
pip install gdown