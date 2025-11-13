# CLAUDE.md - AI Assistant Development Guide

## Project Overview

**inswapper** is a one-click face swapping application powered by [insightface](https://github.com/deepinsight/insightface) with optional face restoration using [CodeFormer](https://github.com/sczhou/CodeFormer). The project performs high-quality face swapping between source and target images, with support for multiple faces and image enhancement.

**Key Capabilities:**
- Single or multiple face swapping
- Face restoration and enhancement
- Background upsampling
- GPU (CUDA) and CPU inference support
- Command-line and Python API interfaces

## Repository Structure

```
inswapper/
├── swapper.py                 # Core face swapping logic (main entry point)
├── restoration.py             # Face restoration using CodeFormer
├── run_in_venv.py            # Virtual environment utilities
├── run.sh                     # Setup and execution script
├── requirements.txt           # Python dependencies
├── inswapper_one_click_face_swapping.ipynb  # Jupyter notebook demo
├── checkpoints/              # Model weights directory (gitignored)
│   └── inswapper_128.onnx   # Main face swap ONNX model (download required)
├── data/                     # Sample images (gitignored in production)
│   ├── man1.jpeg, man2.jpeg  # Source face images
│   ├── mans1.jpeg            # Multi-face target image
│   └── *.png, *.mp4          # Additional test assets
└── CodeFormer/               # Face restoration submodule
    └── CodeFormer/
        ├── basicsr/          # Basic super-resolution library
        ├── facelib/          # Face detection and parsing
        └── weights/          # CodeFormer model weights
```

## Core Components

### 1. swapper.py (14KB)
**Primary face swapping module** - Lines: 322

**Key Functions:**
- `getFaceSwapModel(model_path)` - Load ONNX face swap model
- `getFaceAnalyser(model_path, providers, det_size)` - Initialize face detection
- `get_one_face(face_analyser, frame)` - Detect single face (leftmost)
- `get_many_faces(face_analyser, frame)` - Detect multiple faces (left-to-right order)
- `swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame)` - Perform face swap
- `process(source_img, target_img, model)` - Main processing pipeline

**Important Implementation Details:**
- Face detection uses `buffalo_l` model from insightface
- Faces are sorted left-to-right by bbox[0] (x-coordinate) at swapper.py:49
- Device selection via `DEVICE` environment variable (cuda/cpu) at swapper.py:17
- Supports both single Image and List[Image] as source at swapper.py:200-252

**Command-Line Arguments:**
- `--source_img`: Source image path(s), semicolon-separated for multiple (e.g., "img1.jpg;img2.jpg")
- `--target_img`: Target image path
- `--output_img`: Output path (default: "result.png")
- `--face_restore`: Enable face restoration
- `--background_enhance`: Enhance background quality
- `--face_upsample`: Upsample faces
- `--upscale`: Upscale factor (1-4, default: 1)
- `--codeformer_fidelity`: CodeFormer fidelity (0.0-1.0, default: 0.5)

### 2. restoration.py (7KB)
**Face restoration and enhancement module**

**Key Functions:**
- `check_ckpts()` - Download required CodeFormer weights at restoration.py:23-38
- `set_realesrgan()` - Initialize RealESRGAN upsampler at restoration.py:42-61
- `face_restoration(img, background_enhance, face_upsample, upscale, codeformer_fidelity, upsampler, codeformer_net, device)` - Main restoration pipeline at restoration.py:64-163

**Model Weights (auto-downloaded):**
- `codeformer.pth`: Main restoration model
- `detection_Resnet50_Final.pth`: Face detection
- `parsing_parsenet.pth`: Face parsing
- `RealESRGAN_x2plus.pth`: Super-resolution upsampler

**Important Notes:**
- Adjusts upscale factor based on image size to avoid OOM at restoration.py:78-85
- Uses FaceRestoreHelper with 512x512 face size at restoration.py:87-94
- Supports grayscale and color images
- Device: CUDA > MPS > CPU fallback

### 3. run.sh (3.2KB)
**Automated setup and execution script**

**Workflow:**
1. Install Miniconda (if not present)
2. Create Python 3.7 conda environment
3. Install PyTorch (CUDA 11.8 or CPU-only based on device arg)
4. Install requirements.txt dependencies
5. Download inswapper_128.onnx model
6. Run example face swap operations

**Usage:**
```bash
./run.sh cuda    # GPU mode
./run.sh cpu     # CPU mode
```

## Development Workflows

### Setup (First Time)

**Option 1: Automated (Recommended)**
```bash
chmod +x run.sh
./run.sh cuda  # or cpu
```

**Option 2: Manual**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Download face swap model
mkdir -p checkpoints
wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
  -O checkpoints/inswapper_128.onnx

# Clone CodeFormer (for face restoration)
cd ..
git lfs install
git clone https://huggingface.co/spaces/sczhou/CodeFormer
```

### Running Face Swaps

**Python API (Programmatic):**
```python
from swapper import process
from PIL import Image

# Single source to single/multiple targets
source_img = [Image.open("data/man1.jpeg")]
target_img = Image.open("data/mans1.jpeg")
model = "./checkpoints/inswapper_128.onnx"

result = process(source_img, target_img, model)
result.save("output.png")
```

**Command Line (Production):**
```bash
# Basic swap
python swapper.py \
  --source_img="data/man1.jpeg" \
  --target_img="data/mans1.jpeg" \
  --output_img="result.png"

# With face restoration
python swapper.py \
  --source_img="data/man1.jpeg" \
  --target_img="data/mans1.jpeg" \
  --face_restore \
  --background_enhance \
  --face_upsample \
  --upscale=2 \
  --codeformer_fidelity=0.5
```

**Multiple Source Faces:**
```bash
python swapper.py \
  --source_img="data/man1.jpeg;data/man2.jpeg" \
  --target_img="data/mans1.jpeg" \
  --face_restore
```

### Testing Changes

When modifying the codebase:

1. **Test basic face swapping:**
   ```bash
   python swapper.py --source_img="data/man1.jpeg" --target_img="data/mans1.jpeg"
   ```

2. **Test with restoration (GPU required):**
   ```bash
   python swapper.py \
     --source_img="data/man1.jpeg" \
     --target_img="data/mans1.jpeg" \
     --face_restore --upscale=2
   ```

3. **Test multiple faces:**
   ```bash
   python swapper.py \
     --source_img="data/man1.jpeg;data/man2.jpeg" \
     --target_img="data/mans1.jpeg"
   ```

4. **Verify output images** in the specified output path

## Key Conventions for AI Assistants

### Code Style and Patterns

1. **Face Detection Order:**
   - Faces are ALWAYS sorted left-to-right by x-coordinate (bbox[0])
   - Index 0 = leftmost face, index N-1 = rightmost face
   - See: swapper.py:49 and swapper.py:37-39

2. **Image Format Conventions:**
   - Input: PIL Image objects (RGB)
   - Internal processing: OpenCV format (BGR numpy arrays)
   - Output: PIL Image (RGB)
   - Conversion: `cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)` at swapper.py:214

3. **Device Selection:**
   - Check environment variable `DEVICE` (default: "cuda")
   - Device priority: MPS > CUDA > CPU
   - Set via: `export DEVICE=cuda` or `export DEVICE=cpu`
   - See: swapper.py:17, swapper.py:295

4. **Model Providers:**
   - Use `onnxruntime.get_available_providers()` to auto-detect
   - Common providers: CUDAExecutionProvider, CPUExecutionProvider
   - Set at: swapper.py:204

5. **Error Handling:**
   - Face detection returns None if no faces found (swapper.py:38, 51)
   - Always check for None before processing
   - Graceful degradation: print warnings, continue processing

### File Naming and Paths

1. **Model Checkpoints:**
   - Store in `./checkpoints/` directory
   - Face swap model: `checkpoints/inswapper_128.onnx`
   - CodeFormer weights: `CodeFormer/CodeFormer/weights/`

2. **Input/Output:**
   - Test images in `./data/` directory
   - Default output: `result.png` in project root
   - Support semicolon-separated paths for multiple sources

3. **Gitignore:**
   - Checkpoints, data, venv are gitignored
   - Models must be downloaded separately (not committed)

### Dependencies and Imports

1. **Critical Dependencies:**
   - `insightface>=0.7.3` - Face detection and swapping
   - `onnxruntime-gpu` - GPU inference (or `onnxruntime` for CPU)
   - `torch>=1.7.1` - Deep learning backend
   - `opencv-python` - Image processing
   - `pillow>=9.5.0` - Image I/O

2. **Import Pattern:**
   ```python
   import insightface
   import onnxruntime
   import cv2
   from PIL import Image
   ```

3. **CodeFormer Integration:**
   - Add to sys.path: `sys.path.append('./CodeFormer/CodeFormer')`
   - Import from basicsr and facelib modules
   - See: restoration.py:1-3

### Common Pitfalls and Solutions

1. **Out of Memory (OOM):**
   - Large images auto-reduce upscale at restoration.py:78-85
   - Max upscale: 4x
   - Resolution > 1500px forces upscale=1, disables enhancements

2. **No Faces Detected:**
   - Returns None from face detection functions
   - Check image quality and face visibility
   - Try adjusting det_size parameter (default: 320x320)

3. **Model Download Failures:**
   - Models auto-download on first run
   - Manually verify checksums if corruption suspected
   - Check network connectivity for Hugging Face/GitHub

4. **Device Compatibility:**
   - MPS (Apple Silicon) support via `torch.backends.mps.is_available()`
   - Fall back to CPU if GPU unavailable
   - Use `onnxruntime` (CPU-only) if `onnxruntime-gpu` install fails

### Making Changes

**When modifying swapper.py:**
- Preserve face left-to-right ordering logic
- Maintain BGR/RGB conversion consistency
- Test with both single and multiple source images
- Verify None-handling for edge cases

**When modifying restoration.py:**
- Respect memory constraints (image size checks)
- Ensure model weights auto-download
- Test on both CUDA and CPU devices
- Verify CodeFormer fidelity parameter range (0.0-1.0)

**Adding New Features:**
1. Update argument parser in swapper.py:parse_args()
2. Modify process() function logic
3. Update README.md usage examples
4. Add test cases with sample data
5. Consider backward compatibility

**Git Workflow:**
- Branch naming: `claude/claude-md-*` (session-specific)
- Commit message format: Descriptive, action-oriented
- Push to origin with: `git push -u origin <branch-name>`
- Never force-push to main/master

## Environment Variables

- `DEVICE`: Computation device (cuda/cpu/mps) - Default: "cuda"

## External Resources

- **InswApper Model:** https://huggingface.co/ezioruan/inswapper_128.onnx
- **CodeFormer:** https://huggingface.co/spaces/sczhou/CodeFormer
- **Insightface:** https://github.com/deepinsight/insightface
- **Original Inspiration:** https://github.com/s0md3v/sd-webui-roop

## Troubleshooting

**Issue: CUDA out of memory**
- Solution: Use CPU mode via `export DEVICE=cpu`
- Or reduce image resolution before processing

**Issue: Face not detected**
- Solution: Ensure face is clearly visible and frontal
- Try increasing detection size: modify `det_size=(640, 640)` in getFaceAnalyser()

**Issue: Poor restoration quality**
- Solution: Adjust `--codeformer_fidelity` (higher = more faithful to original, lower = more restoration)
- Try different upscale values (1-4)

**Issue: Model download fails**
- Solution: Manually download from URLs in restoration.py:24-28 and run.sh:56
- Place in correct directories as specified

## Performance Considerations

1. **GPU Acceleration:**
   - 10-100x faster than CPU for face restoration
   - Requires CUDA-compatible GPU and onnxruntime-gpu

2. **Batch Processing:**
   - Current implementation processes one image at a time
   - For bulk processing, wrap in loop over image directory

3. **Memory Usage:**
   - ~2-4GB GPU memory for typical 512x512 faces
   - Scales with image resolution and upscale factor
   - Auto-adjustment at restoration.py:78-85 prevents most OOM errors

---

**Last Updated:** 2025-11-13
**Project Version:** Based on commit 5d68591
**Maintained by:** AI assistants using this guide should keep it current with codebase changes
