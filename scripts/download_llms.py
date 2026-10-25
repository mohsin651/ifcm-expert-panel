"""Download the ten panel LLMs into a dedicated local Hugging Face cache.

Run with the upstream virtual environment. The cache can be reused by setting
HF_HUB_CACHE to the same directory before generating votes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import model_info, snapshot_download


MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "aaditya/Llama3-OpenBioLLM-8B",
    "m42-health/Llama3-Med42-8B",
    "01-ai/Yi-1.5-9B-Chat",
    "ibm-granite/granite-3.1-8b-instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="dedicated local cache with at least 160 GB free")
    parser.add_argument("--manifest", type=Path, default=Path("fresh_runs/models_manifest.json"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.cache).free < 160 * 10**9:
        raise SystemExit("Need at least 160 GB free on the chosen cache filesystem")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    for index, name in enumerate(MODELS, 1):
        if name in manifest:
            print(f"[{index}/10] already recorded: {name}", flush=True)
            continue
        info = model_info(name)
        patterns = ["*.json", "*.py", "*.model", "*.txt", "*.tiktoken", "*.safetensors"]
        if name == "aaditya/Llama3-OpenBioLLM-8B":
            patterns.append("pytorch_model-*.bin")
        print(f"[{index}/10] downloading {name} at revision {info.sha}", flush=True)
        path = snapshot_download(
            repo_id=name,
            revision=info.sha,
            cache_dir=str(args.cache),
            allow_patterns=patterns,
            ignore_patterns=["consolidated.safetensors", "original/*"],
            max_workers=4,
        )
        manifest[name] = {"revision": info.sha, "snapshot": path}
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[{index}/10] complete: {name}", flush=True)


if __name__ == "__main__":
    main()
