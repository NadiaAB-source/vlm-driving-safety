# Safety-Aware Post-Processing for Reliable VLM Driving Decisions

## 📌 Project Description

This project improves the safety and reliability of Vision–Language Models (VLMs) for autonomous driving without retraining.

Modern VLMs can analyze driving scenes and generate decisions such as *keep speed* or *brake*. However, they may produce unsafe outputs (e.g., accelerating at a red light).

We propose a **post-processing safety layer** that operates on top of a pretrained model and:
- Converts outputs into structured actions
- Applies deterministic safety rules
- Uses multi-query consistency voting

This method is:
- Model-agnostic  
- Explainable  
- Does not require retraining  

---

## ⚙️ Software Requirements

- Python 3.10+
- PyTorch
- HuggingFace Transformers

Install dependencies using:

```bash
pip install -r requirements.txt