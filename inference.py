# ============================================================
# IMPORTS
# ============================================================

import os
import json
import re
import torch

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ============================================================
# LOAD QWEN2-VL MODEL
# ============================================================

def load_model():
    """
    Load Qwen2-VL model and processor.

    Returns:
        model, processor
    """

    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return model, processor


# ============================================================
# PROMPTS
# ============================================================

ACTION_SYSTEM_PROMPT = """You are an expert autonomous driving assistant analyzing a real dashcam image.
Study the scene carefully: vehicles, pedestrians, cyclists, traffic lights, road signs, lane markings.
Answer the planning question about what the ego vehicle should do.
Respond ONLY in this exact JSON format, no other text:
{"action": "<specific driving action>", "reason": "<max 5 words>"}
Action must be one of: keep speed, accelerate, brake gently, brake hard, stop, turn left, turn right, change lane left, change lane right, back up"""


CONTEXT_SYSTEM_PROMPT = """You are analyzing a dashcam image for an autonomous driving safety system.
Answer each question with ONLY the exact option given. Be precise and conservative."""


CONTEXT_USER_PROMPT = """Carefully examine this dashcam image and answer each question:

1. Is there a traffic light visible? If yes, what color is it? Answer: red / yellow / green / none
2. Is there a stop sign visible? Answer: yes / no
3. Is there a crosswalk visible? Answer: yes / no
4. Is there a pedestrian or cyclist visible in or near the road? Answer: yes / no / unsure
5. Is there a vehicle directly ahead within about 2 car lengths? Answer: yes / no / unsure
6. Is there a vehicle directly behind within about 2 car lengths? Answer: yes / no / unsure
7. Is the lane ahead blocked? Answer: yes / no / unsure
8. Is there space to change lane to the left? Answer: yes / no / unsure
9. Is there space to change lane to the right? Answer: yes / no / unsure
10. Is visibility degraded (brightness, fog, glare, night)? Answer: yes / no

Now respond ONLY with a JSON object using these exact keys:
{"traffic_light": "?", "stop_sign": "?", "crosswalk": "?", "pedestrian": "?", "vehicle_ahead": "?", "vehicle_behind": "?", "lane_blocked": "?", "drivable_left": "?", "drivable_right": "?", "visibility_degraded": "?"}
"""


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def run_qwen(model, processor, image_path, system_prompt, user_text,
             temperature=0.2, max_new_tokens=256):
    """
    Run Qwen2-VL on image + prompt.

    Args:
        model: loaded VLM model
        processor: tokenizer/processor
        image_path (str): path to image
        system_prompt (str)
        user_text (str)

    Returns:
        str: raw model output
    """

    image_uri = "file://" + os.path.abspath(image_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image_uri},
            {"type": "text", "text": user_text},
        ]},
    ]

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    # Move to GPU if available
    inputs = {
        k: v.to("cuda")
        for k, v in inputs.items()
        if hasattr(v, "to")
    }

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=(temperature > 0)
        )

    out_tokens = gen[:, inputs["input_ids"].shape[1]:]

    return processor.batch_decode(out_tokens, skip_special_tokens=True)[0]


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_output(raw):
    """
    Extract JSON safely from model output.

    Handles:
    - clean JSON
    - JSON inside text
    - malformed outputs

    Returns:
        dict
    """

    raw = raw.strip()

    # Try direct parsing
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to extract JSON block
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return {"parse_error": True, "raw": raw}