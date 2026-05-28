"""Debug: see what the processor actually does with a single audio."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_env_lib_bin = Path(sys.executable).parent / "Library" / "bin"
if _env_lib_bin.exists():
    os.environ["PATH"] = str(_env_lib_bin) + os.pathsep + os.environ.get("PATH", "")

import librosa
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2AudioForConditionalGeneration,
)

QWEN_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
SR = 16000
PILOT_ROOT = Path(__file__).resolve().parent.parent

sample_path = PILOT_ROOT / "data" / "pilot_samples" / "late_suffix_00.wav"
audio, _ = librosa.load(str(sample_path), sr=SR, mono=True)

processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
print("processor type:", type(processor).__name__)
print("feature_extractor sr:", processor.feature_extractor.sampling_rate)

# Build conversation. Try the modern API first.
INSTRUCTION = "Listen carefully to the audio. Respond appropriately to whatever the speaker is asking or telling you."
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": str(sample_path)},
            {"type": "text", "text": INSTRUCTION},
        ],
    }
]
chat_text = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=False
)
print("\n=== rendered chat_text ===")
print(repr(chat_text))

# Process
inputs = processor(text=chat_text, audio=audio.astype(np.float32), return_tensors="pt", padding=True)
print("\n=== inputs keys & shapes ===")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__}")

# Decode the input_ids to see what tokens went in
print("\n=== decoded input_ids ===")
print(repr(processor.tokenizer.decode(inputs["input_ids"][0])))

# Quick generation
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
print("\n=== loading model ===")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    QWEN_MODEL_ID, quantization_config=bnb, device_map="auto",
    max_memory={0: "7GiB", "cpu": "32GiB"},
)
model.eval()
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
prompt_len = inputs["input_ids"].shape[1]
text = processor.batch_decode(ids[:, prompt_len:], skip_special_tokens=True)[0]
print("\n=== output ===")
print(repr(text))
