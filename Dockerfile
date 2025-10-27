# Use a slim Python 3.10 base image
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install PyTorch for CPU first (critical for non-GPU environments)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy the rest of your project code into the container
COPY . .

# Download the production models into the image
# This bundles the models, so you don't download them every time you run
RUN mkdir -p weights && \
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt -O weights/yolov8l.pt && \
    wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth -O weights/sam_hq_vit_l.pth

# Set the entrypoint to run your script
ENTRYPOINT ["python", "advanced_annotater.py"]