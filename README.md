# Safety-Aware Post-Processing for Reliable VLM Driving Decisions
This project presents a safety-aware post-processing framework designed to improve the reliability of Vision–Language Models (VLMs) in autonomous driving scenarios.

Instead of modifying or retraining the model, we introduce a lightweight, model-agnostic layer that:
- Extracts structured scene context
- Applies deterministic safety rules to filter unsafe actions
- Uses multi-query consistency voting to stabilize predictions

The system significantly reduces unsafe decisions while maintaining or improving overall accuracy, making it suitable for safety-critical applications.

Key features:
- No model retraining required
- Fully explainable rule-based decisions
- Works with any Vision–Language Model
- Supports both lightweight (dummy) and full dataset execution modes
## Project Description

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

## Software Requirements

- Python 3.10+
- PyTorch
- HuggingFace Transformers

Install dependencies using:

```bash
pip install -r requirements.txt
