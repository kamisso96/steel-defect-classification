# Steel Surface Defect Classification using Deep Learning

Master's thesis project at Anhui University of Technology.

## Overview
This project systematically compares attention mechanisms, transfer learning, and transformer architectures for steel surface defect classification on the NEU dataset under CPU‑constrained conditions. A novel MobileNetV2 + SE model achieves **99.07%** accuracy.

## Models
- Baseline CNN
- CNN + SE (Squeeze-and-Excitation)
- CNN + CBAM
- MobileNetV2 (transfer learning)
- TinyViT (Vision Transformer)
- Hybrid CNN‑Transformer
- **Proposed: MobileNetV2 + SE**

## Results
| Model | Accuracy |
|-------|----------|
| MobileNetV2 + SE (proposed) | **99.07%** |
| MobileNetV2 | 97.69% |
| ... | ... |

## Repository Structure
- `data/` – dataset (excluded from repo; instructions to download provided)
- `models/` – model definitions
- `results/` – saved models and test results
- `train.py` – training script
- `generate_plots.py` – figure generation
- `thesis/` – LaTeX/Word source of the thesis (if included)

## Requirements
- Python 3.10
- TensorFlow 2.16.1
- See `requirements.txt`

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the NEU dataset and place it in `data/NEU/`.
4. Run `python train.py --model <model_name>`.
5. Run `python generate_plots.py` to reproduce figures.

## Author
DIAKITE DRAMANE KAMISSO – Anhui University of Technology, 2026

## License
MIT (or choose another)
