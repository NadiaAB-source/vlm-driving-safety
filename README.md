# Safety-Aware Post-Processing for Reliable VLM Driving Decisions

## Project Overview
This project presents a safety-aware post-processing framework that improves the reliability of Vision–Language Models (VLMs) for autonomous driving decision-making.

VLMs can analyze driving scenes and generate high-level actions such as keep speed or brake gently. However, these models are:
- stochastic (inconsistent across runs)
- not safety-aware (may violate traffic rules)

To address this, we introduce a lightweight post-processing pipeline that:
- extracts structured scene context
- applies deterministic safety rules
- performs multi-run consistency voting

This improves both safety and accuracy without retraining the model.

## Key Features
- No model retraining required  
- Model-agnostic (works with any VLM)  
- Fully explainable rule-based decisions  
- Multi-query consistency (K=3)  
- Error analysis and visualization  
- Separate demo script for qualitative evaluation  

## Hardware Requirements
- GPU recommended: NVIDIA A100 / RTX 3090 (≥16GB VRAM)
- Minimum: GPU with ≥12GB VRAM
- CPU-only execution is possible but very slow

## Software Requirements
- Python 3.10+
- PyTorch
- HuggingFace Transformers
- datasets

Install dependencies:
pip install -r requirements.txt

## Project Structure
vlm-driving-safety/
├── main.py              # Full evaluation pipeline  
├── demo.py              # Qualitative demo (single sample)  
├── inference.py         # Model loading + inference  
├── safety_rules.py      # Rule-based safety logic  
├── consistency.py       # Majority voting  
├── utils.py             # Action normalization  
├── metrics.py           # Evaluation + plots  
├── data/DriveBench/     # Dataset (images only)  
├── results/             # Output JSON + plots  
├── requirements.txt  
├── README.md  

## Data Sources
- Questions / metadata → HuggingFace  
- Images → Local dataset folder  

## Dataset Setup
Step 1: Dataset metadata is loaded automatically using:
load_dataset("drive-bench/arena")

Step 2: Download DriveBench images manually and place them in:
data/DriveBench/Brightness/

Expected structure:
data/DriveBench/Brightness/
├── CAM_FRONT/  
├── CAM_BACK/  
├── CAM_FRONT_LEFT/  
├── CAM_FRONT_RIGHT/  
├── CAM_BACK_LEFT/  
├── CAM_BACK_RIGHT/  

Images are not included in the repository due to size constraints.

## How to Run

Full Evaluation Pipeline:
python main.py

This will:
- load dataset from HuggingFace
- run VLM inference (K=3)
- apply safety rules
- compute metrics
- generate plots
- save results

## Output Files
Saved in:
results/

Includes:
- vlm_results.json → full results  
- vlm_results_chart.png → metrics plot  
- vlm_error_categorization.png → error analysis  

## Qualitative Demo
Run:
python demo.py

This runs one sample and shows:
- Question  
- Qwen raw outputs (K=3)  
- Image description (context)  
- Safety rules applied  
- Final decision  
- Summary  

## Pipeline Overview
Image + Question  
→ Qwen2-VL Model (K=3 runs)  
→ Context Extraction  
→ Safety Rules Layer  
→ Consistency Voting  
→ Final Safe Decision  

## Evaluation Metrics
- Decision Accuracy  
- Unsafe Decision Rate  
- False Override Rate  
- Consistency Score  

## Error Analysis
Errors are categorized into:
- Context Extraction Errors  
- Rule Conflicts (false overrides)  
- Ambiguous Scenes  

Example output:
{
  "baseline_final": "accelerate",
  "safe_final": "keep speed",
  "any_override": true,
  "fired_rules": ["R8"]
}

## Sample Results
- Unsafe Rate reduced: 30.5% → 1.5%  
- Accuracy improved: 43.3% → 57.9%  
- False override rate: 2.8%  

## Reproducibility
To reproduce:
- Install dependencies  
- Place images in data/DriveBench/  
- Run:
  python main.py  
  python demo.py  

## Limitations
- Depends on VLM context extraction quality  
- Rule-based system is not fully scalable  
- Single-frame only (no temporal reasoning)  
- Dataset limited to ~400 samples  

## Authors
- Nadia Badawi  
- Sema Helali  
- Kisaa Fatima Muhammad  
- Hailemariam Teshager  
