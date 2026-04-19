# Safety-Aware Post-Processing for Reliable VLM Driving Decisions

## Project Description

This project presents a **safety-aware post-processing framework** designed to improve the reliability of Vision–Language Models (VLMs) in autonomous driving scenarios.

Modern VLMs can analyze driving scenes and generate high-level decisions such as *keep speed* or *brake gently*. However, these models are inherently stochastic and may produce unsafe outputs (e.g., accelerating at a red light).

To address this, we introduce a **lightweight, model-agnostic safety layer** that operates on top of a pretrained VLM without retraining.

The framework:
- Extracts structured scene context from images
- Applies deterministic safety rules to enforce traffic constraints
- Uses multi-query consistency voting to stabilize predictions

This approach significantly reduces unsafe decisions while improving overall accuracy.

---

## Key Features

- No model retraining required  
- Model-agnostic (works with any VLM)  
- Fully explainable rule-based decisions  
- Multi-query consistency for stability  
- Supports both lightweight testing and full dataset evaluation  

---

## Hardware Requirements

- GPU recommended: NVIDIA A100 / RTX 3090 (16GB+ VRAM)
- Minimum: GPU with 12GB VRAM
- CPU-only execution is possible but extremely slow

---

## Software Requirements

- Python 3.10+
- PyTorch
- HuggingFace Transformers
- Datasets library

---

## Installation

```bash
git clone https://github.com/NadiaAB-source/vlm-driving-safety.git
cd vlm-driving-safety

pip install -r requirements.txt
