import os, sys
from pathlib import Path
_env_lib_bin = Path(sys.executable).parent / "Library" / "bin"
if _env_lib_bin.exists():
    os.environ["PATH"] = str(_env_lib_bin) + os.pathsep + os.environ.get("PATH", "")

import librosa, numpy as np, torch
from pyannote.audio import Pipeline

PILOT_ROOT = Path(__file__).resolve().parent.parent
sample = PILOT_ROOT / "data" / "pilot_samples" / "second_speaker_00.wav"

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

audio, _ = librosa.load(str(sample), sr=16000, mono=True)
wf = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
out = pipeline({"waveform": wf, "sample_rate": 16000})

print("type:", type(out).__name__)
print("dir:", [a for a in dir(out) if not a.startswith("_")])
print("repr:", repr(out)[:500])
# Try a few common access patterns
for attr in ("speaker_diarization", "diarization", "segments", "annotation"):
    if hasattr(out, attr):
        v = getattr(out, attr)
        print(f"  out.{attr}: type={type(v).__name__}")
        if hasattr(v, "itertracks"):
            print(f"    -> has itertracks!")
            for turn, _, spk in v.itertracks(yield_label=True):
                print(f"    turn {turn.start:.2f}-{turn.end:.2f} spk={spk}")
            break
