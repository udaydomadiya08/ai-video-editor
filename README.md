# AI Video Editing Style Learning System

Professional AI system that learns video editing styles from reference videos and applies them to new content.

## Features

- **Incremental Learning**: Resume training with `--resume` flag to build on existing models
- **GPU Acceleration**: Supports NVIDIA CUDA, Apple Silicon MPS, and CPU training
- **Multi-GPU Training**: Train across multiple GPUs for faster results
- **Pure Neural Learning**: No hardcoded rules - learns editing patterns from data
- **Style Transfer**: Applies learned editing styles to new videos

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/udaydomadiya08/ai-video-editor.git
cd ai-video-editor
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python main.py --help
```

## Quick Start

### Training

Train the AI on your reference videos:

```bash
# Basic training (auto-detect device)
python main.py train --refs ./data --output ./models

# Force CPU
python main.py train --refs ./data --output ./models --cpu

# Use Apple Silicon GPU
python main.py train --refs ./data --output ./models --mps

# Use single NVIDIA GPU
python main.py train --refs ./data --output ./models --gpu

# Use 2 GPUs
python main.py train --refs ./data --output ./models --gpu 2

# Use all available GPUs
python main.py train --refs ./data --output ./models --gpu -1

# Resume training with new data
python main.py train --refs ./new_data --output ./models --resume --gpu -1
```

### Generate Video

Apply learned style to new content:

```bash
python main.py generate --input video.mp4 --music track.mp3 --output output.mp4 --models ./models
```

## Project Structure

```
learn_fictic/
├── main.py                 # Main CLI entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── modules/               # Core modules
│   ├── style_learner.py          # Learn editing patterns
│   ├── neural_transition_vae.py  # Learn cut dynamics
│   ├── neural_style_vae.py       # Learn visual style
│   ├── neural_effect_vae.py      # Learn effects
│   ├── music_analyzer.py         # Music analysis
│   ├── auto_editor.py            # Auto video editing
│   └── ...
├── utils/                 # Utility functions
│   ├── video_io.py
│   └── audio_io.py
├── data/                  # Training videos (create this)
└── models/                # Trained models (created automatically)
```

## Training Data

Place your reference edited videos in the `./data` directory:

```bash
mkdir -p data
# Copy your reference videos to ./data/
```

The AI will learn from these videos and apply the learned style to new content.

## System Requirements

### Minimum
- **CPU**: Multi-core processor (Intel/AMD/Apple Silicon)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.8 or higher

### Recommended for Fast Training
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **RAM**: 16GB+
- **Storage**: SSD with 20GB+ free space

### Apple Silicon (M1/M2/M3)
- Supports MPS (Metal Performance Shaders) for GPU acceleration
- 8GB unified memory minimum, 16GB+ recommended

## Performance

Training speed comparison (500 sequences, 20 epochs):

| Device | Time |
|--------|------|
| CPU (M1) | ~1h05m |
| MPS (M1) | ~6-13m |
| 1 GPU (RTX 3090) | ~40s-1.3m |
| 2 GPUs (RTX 3090) | ~22-45s |
| 4 GPUs (RTX 3090) | ~11-24s |

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors during training:

1. The system automatically limits sequences to 500 for M1 8GB RAM
2. For other systems, you can reduce batch sizes in the config
3. Use `--cpu` flag to force CPU training (slower but more stable)

### CUDA Not Available

If you have an NVIDIA GPU but CUDA is not detected:

1. Install CUDA toolkit from NVIDIA website
2. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### MPS Not Available

If you have Apple Silicon but MPS is not detected:

1. Update to macOS 12.3 or later
2. Update to Python 3.8 or later
3. Update PyTorch: `pip install --upgrade torch`

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]

## Support

For issues and questions, please open an issue on GitHub.
