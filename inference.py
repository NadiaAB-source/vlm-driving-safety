import os
import json
import re
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
qwen_model.eval()

qwen_processor = AutoProcessor.from_pretrained(MODEL_ID)


ACTION_SYSTEM_PROMPT = """You are an expert autonomous driving assistant analyzing a real dashcam image.
Respond ONLY in JSON:
{"action": "<action>", "reason": "<short reason>"}"""

CONTEXT_USER_PROMPT = """Carefully examine the image and answer:
traffic_light: red/yellow/green/none
stop_sign: yes/no
crosswalk: yes/no
pedestrian: yes/no/unsure
vehicle_ahead: yes/no/unsure
vehicle_behind: yes/no/unsure
lane_blocked: yes/no/unsure
drivable_left: yes/no/unsure
drivable_right: yes/no/unsure
visibility_degraded: yes/no

Return JSON only."""



def run_qwen(image_path, system_prompt, user_text, temperature=0.2, max_new_tokens=256):
    image_uri = "file://" + os.path.abspath(image_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image_uri},
            {"type": "text", "text": user_text},
        ]},
    ]

    text_input = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = qwen_processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    )

    inputs = {k: v.to("cuda") for k, v in inputs.items() if hasattr(v, "to")}

    with torch.no_grad():
        gen = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=(temperature > 0)
        )

    out_tokens = gen[:, inputs["input_ids"].shape[1]:]

    return qwen_processor.batch_decode(out_tokens, skip_special_tokens=True)[0]


  


def parse_json_output(raw):
    raw = raw.strip()

    try:
        return json.loads(raw)
    except:
        pass

    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return {"parse_error": True, "raw": raw}