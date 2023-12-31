#!/bin/bash

# Step 1: Clone the GitHub repository
#git clone https://github.com/CesarChaMal/inswapper.git
#cd inswapper

# Step 2: Install dependencies
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# Step 3: Download the model and move it to the 'models' directory
#wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O inswapper_128.onnx
curl -L https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -o inswapper_128.onnx

mkdir -p checkpoints
mv inswapper_128.onnx ./checkpoints

# Step 4: Run the Python script
# Replace '/content/video.mp4' and '/content/image.jpeg' with your actual file paths
python swapper.py --source_img data/source_image1.png --target_img data/target_image.png --output_img /content/output_image.jpg

