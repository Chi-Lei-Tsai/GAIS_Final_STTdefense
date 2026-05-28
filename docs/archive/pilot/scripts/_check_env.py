import importlib
mods = ['torch','torchaudio','librosa','soundfile','numpy','tqdm','edge_tts','transformers','accelerate','bitsandbytes','sentence_transformers','pyannote.audio']
for m in mods:
    try:
        x = importlib.import_module(m)
        v = getattr(x, '__version__', '?')
        print(f'OK      {m} {v}')
    except Exception as e:
        print(f'MISSING {m} -- {type(e).__name__}: {e}')
