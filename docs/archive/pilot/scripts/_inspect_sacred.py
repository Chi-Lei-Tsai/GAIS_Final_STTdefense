"""List files in SACRED-Bench repo to find direct download paths."""
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("tsinghua-ee/SACRED-Bench", repo_type="dataset")
by_top = {}
for f in files:
    top = f.split("/")[0]
    by_top.setdefault(top, []).append(f)
for top, lst in sorted(by_top.items()):
    print(f"=== {top} ({len(lst)} files) ===")
    for f in lst[:6]:
        print(f"  {f}")
    if len(lst) > 6:
        print(f"  ... +{len(lst) - 6} more")
