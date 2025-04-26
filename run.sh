#!/bin/bash
set -e

# Check for user input
DEVICE=${1:-cuda} # default to cuda if not provided

if [[ "$DEVICE" != "cuda" && "$DEVICE" != "cpu" ]]; then
  echo "Invalid device option. Use: cuda or cpu"
  exit 1
fi

echo "Running setup for device: $DEVICE"

# Step 1: Install Miniconda if not already installed
if [ ! -d "$HOME/miniconda3" ]; then
  echo "Miniconda not found. Downloading and installing Miniconda locally..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b -p $HOME/miniconda3
else
  echo "Miniconda already installed at $HOME/miniconda3."
fi

# Step 2: Initialize conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# Step 3: Create and activate virtual environment if not already existing
if ! conda info --envs | grep -q "^venv"; then
  echo "Creating virtual environment 'venv' with Python 3.7..."
  conda create --name venv python=3.7 -y
else
  echo "Virtual environment 'venv' already exists."
fi

conda activate venv

# Step 4: Install torch based on device
if [ "$DEVICE" == "cuda" ]; then
  echo "Installing GPU-accelerated PyTorch (CUDA 11.8)..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
  echo "Installing CPU-only PyTorch..."
  pip install torch torchvision torchaudio
fi

# Step 5: Install required Python packages
echo "Installing Python packages from requirements.txt..."
pip install -r requirements.txt

# Step 6: Download inswapper model if not already downloaded
mkdir -p checkpoints
if [ ! -f "checkpoints/inswapper_128.onnx" ]; then
  echo "Downloading inswapper_128.onnx model..."
  wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O checkpoints/inswapper_128.onnx
else
  echo "inswapper_128.onnx model already exists."
fi

# Step 7: Verify installation
echo "Installed Python packages:"
pip list

# Step 8: Run face swaps
echo "Running face swap examples on device: $DEVICE..."

# Set device environment variable (pass to python)
export DEVICE=$DEVICE

# (1) Single source and target (first run)
python swapper.py --source_img data/source_image.png --target_img data/target_image.png --output_img data/output_image1.jpg --face_restore --background_enhance --face_upsample --upscale=2 --codeformer_fidelity=0.5

# (2) Single source and target (second run)
python swapper.py --source_img data/source_image1.png --target_img data/target_image.png --output_img data/output_image2.jpg --face_restore --background_enhance --face_upsample --upscale=2 --codeformer_fidelity=0.5

# (3) Multiple source images to one target (first run)
python swapper.py --source_img "data/man1.jpeg;data/man2.jpeg" --target_img data/mans1.jpeg --output_img data/output_image3.jpg --face_restore --background_enhance --face_upsample --upscale=2 --codeformer_fidelity=0.5

# (4) Multiple source images to one target (second run)
python swapper.py --source_img "data/man3.jpg;data/man4.jpg" --target_img data/mans1.jpeg --output_img data/output_image4.jpg --face_restore --background_enhance --face_upsample --upscale=2 --codeformer_fidelity=0.5

echo "All swaps completed!"
