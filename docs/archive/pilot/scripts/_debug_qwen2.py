"""Find the correct way to feed audio to Qwen2AudioProcessor in transformers 5.x."""
import inspect
import os
import sys
from pathlib import Path
_env_lib_bin = Path(sys.executable).parent / "Library" / "bin"
if _env_lib_bin.exists():
    os.environ["PATH"] = str(_env_lib_bin) + os.pathsep + os.environ.get("PATH", "")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

# Inspect __call__ signature
sig = inspect.signature(processor.__call__)
print("processor.__call__ signature:")
for name, p in sig.parameters.items():
    print(f"  {name}: default={p.default}")

# Inspect apply_chat_template signature
sig2 = inspect.signature(processor.apply_chat_template)
print("\nprocessor.apply_chat_template signature:")
for name, p in sig2.parameters.items():
    print(f"  {name}: default={p.default}")

# Check the docstring for hints
print("\nprocessor.__call__ docstring (first 1500 chars):")
doc = processor.__call__.__doc__ or ""
print(doc[:1500])
