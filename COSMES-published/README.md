# COSMES: A Comprehensive Framework for Reliable Spatial Intelligence in Multimodal Large Language Models

Official implementation of COSMES, an inference-time framework designed to improve geometric reliability without modifying backbone weights.

## Overview

COSMES (Core Spatial Intelligence and Reasoning) combines:
- **Depth-Aware Frame Selection**: replacing pixel-level L1 difference with a two-component geometric score
- **Semantic-Geometric Dual-Branch Aggregation**: running inference with semantically and geometrically focused prompts
- **Spatial Chain-of-Thought Prompting**: enforcing explicit three-stage spatial reasoning (Identify, Quantify, Verify)
- **Geometric Self-Correction**: a lightweight post-hoc module enforcing domain-specific spatial priors
- **Reliability Assessment**: joint hallucination detection and cross-task consistency verification
- **Bayesian Uncertainty Quantification**: via high-temperature sampling and t-distribution confidence intervals

## Model Download

COSMES uses **Spatial-MLLM** as its backbone, which integrates a VGGT spatial encoder with Qwen2.5-VL for spatial reasoning tasks.

Download the model from Hugging Face:

| Model | Hugging Face | Description |
|-------|-------------|-------------|
| Spatial-MLLM-subset-sft | [Diankun/Spatial-MLLM-subset-sft](https://huggingface.co/Diankun/Spatial-MLLM-subset-sft) | Default backend with VGGT spatial encoder |

### Download via CLI

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download the model to local cache
huggingface-cli download Diankun/Spatial-MLLM-subset-sft
```

### Download via Python

```python
from huggingface_hub import snapshot_download

snapshot_download("Diankun/Spatial-MLLM-subset-sft")
```

The model will be cached at `~/.cache/huggingface/hub/` and automatically discovered by COSMES at runtime.

## Installation

```bash
# 1. Install PyTorch (choose the version matching your CUDA)
# GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install other dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Fastest verification (CPU, 1 question, 4 frames)
python run_spatial_inference.py --mode cpu --quick

# CPU, all 3 questions
python run_spatial_inference.py --mode cpu

# GPU, all 3 questions
python run_spatial_inference.py --mode gpu
```

## Backend Selection

COSMES supports two backends for the Spatial-MLLM model:

```bash
# Default: Spatial-MLLM + VGGT spatial encoder (recommended)
python run_spatial_inference.py --mode cpu

# Ablation: pure Qwen2.5-VL without VGGT
python run_spatial_inference.py --mode cpu --no-vggt
```

## Innovation Modules

### Depth-Aware Frame Selection (`--depth-aware`)
Uses Laplacian variance (geometric depth proxy) + inter-frame normalized cross-correlation (viewpoint change proxy) to jointly drive frame selection.

```bash
python run_spatial_inference.py --mode cpu --depth-aware
```

### Semantic-Geometric Dual-Branch Aggregation (`--dual-branch`)
Runs inference with semantically and geometrically focused prompts separately, averaging numeric answers and selecting the more complete descriptive answer.

```bash
python run_spatial_inference.py --mode cpu --dual-branch
```

### Spatial Chain-of-Thought (`--spatial-cot`)
Replaces standard questions with CoT versions (Identify → Quantify → Verify) for step-by-step spatial reasoning.

```bash
python run_spatial_inference.py --mode cpu --spatial-cot
```

### Geometric Self-Correction + Hallucination Detection + Cross-Task Consistency
```bash
python run_spatial_inference.py --mode cpu \
    --geo-correct \
    --hallucination-check \
    --cross-validate
```
- `--geo-correct`: auto-correct count/distance results based on spatial priors
- `--hallucination-check`: detect unreasonable outputs (negative counts, zero distances, etc.)
- `--cross-validate`: cross-check logical consistency among count, layout, and distance answers

### Bayesian Uncertainty Quantification (`--uncertainty N`)
Appends N high-temperature (T=0.7) samples, reporting mean ± std with confidence levels.

```bash
python run_spatial_inference.py --mode cpu --uncertainty 2
```

### Edge Enhancement Preprocessing (`--edge-enhance`)
Overlays Laplacian edge maps or CLAHE contrast enhancement on input frames.

```bash
python run_spatial_inference.py --mode cpu --edge-enhance
```

## Project Structure

```
COSMES-published/
├── run_spatial_inference.py   # Main entry point
├── space_aware_frames.py      # Frame sampling strategies
├── preprocessing.py           # Input preprocessing (edge enhancement / CLAHE)
├── postprocessing.py          # Post-processing (geo-correction / hallucination / uncertainty)
├── requirements.txt           # Dependencies
├── assets/
│   └── arkitscenes_41069025.mp4   # Demo video
└── src/
    └── models/                    # Spatial-MLLM model code (including VGGT)
```

## Results

COSMES improves distance-related metrics and overall spatial score relative to the Spatial-MLLM baseline, with the largest gains observed in absolute distance error reduction.

| Method | mean_mra | mean_all | Distance Error (m) |
|--------|----------|----------|--------------------|
| Spatial-MLLM | 0.563 | 0.281 | 1.40 |
| COSMES (ours) | 0.581 | 0.290 | 1.30 |
| COSMES + All (ours) | **0.853** | **0.427** | **0.31** |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use COSMES in your research, please cite our work:

```
@article{lu2025cosmes,
  title={COSMES: A Comprehensive Framework for Reliable Spatial Intelligence in Multimodal Large Language Models},
  author={Lu, XingGuang and Kang, Liang},
  journal={Under review},
  year={2025}
}
```
