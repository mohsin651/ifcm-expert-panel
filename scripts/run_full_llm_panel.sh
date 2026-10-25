#!/usr/bin/env bash
# Run all 10 final LLMs on Derm7pt and LIDC, keeping archive CSVs untouched.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${IFCM_PYTHON:-$PWD/.venv-upstream/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python environment: $PYTHON (set IFCM_PYTHON to the GPU Python executable)" >&2
  exit 1
fi
export HF_HOME=/var/tmp/ifcm-llm-cache-mimam
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IFCM_MODEL_MANIFEST="$PWD/fresh_runs/models_manifest.json"
export PYTHONUNBUFFERED=1

"$PYTHON" - "$IFCM_MODEL_MANIFEST" <<'PY'
import json
import pathlib
import sys
models = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert len(models) == 10, f"Expected 10 local models; found {len(models)}"
for name, info in models.items():
    path = pathlib.Path(info["snapshot"])
    assert path.is_dir() and (path / "config.json").is_file(), f"Missing model files: {name}: {path}"
print("All 10 local model snapshots are visible.")
PY

mkdir -p fresh_runs/full_panel/derm fresh_runs/full_panel/lidc

run_job() {
  local job="$1" part="$2" out="$3" models="$4"
  echo "Starting $job part $part: $models"
  IFCM_BUILD_ONLY=1 ./kaggle/push.sh "$job" full "$part"
  IFCM_OUT="$PWD/fresh_runs/full_panel/$out" IFCM_MODELS="$models" \
    "$PYTHON" "kaggle/build/ifcm-$job-full${part:+-$part}/run.py"
}

run_job llm-panel '' derm 'Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,allenai/OLMo-2-1124-7B-Instruct,microsoft/Phi-3.5-mini-instruct'
run_job llm-panel2 '' derm 'aaditya/Llama3-OpenBioLLM-8B,m42-health/Llama3-Med42-8B,01-ai/Yi-1.5-9B-Chat,ibm-granite/granite-3.1-8b-instruct,tiiuae/Falcon3-7B-Instruct,microsoft/Phi-4-mini-instruct'
run_job llm-panel3 0 derm 'Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,allenai/OLMo-2-1124-7B-Instruct,microsoft/Phi-3.5-mini-instruct,aaditya/Llama3-OpenBioLLM-8B'
run_job llm-panel3 1 derm 'm42-health/Llama3-Med42-8B,01-ai/Yi-1.5-9B-Chat,ibm-granite/granite-3.1-8b-instruct,tiiuae/Falcon3-7B-Instruct,microsoft/Phi-4-mini-instruct'
run_job llm-lidc 0 lidc 'Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,allenai/OLMo-2-1124-7B-Instruct,microsoft/Phi-3.5-mini-instruct,aaditya/Llama3-OpenBioLLM-8B'
run_job llm-lidc 1 lidc 'm42-health/Llama3-Med42-8B,01-ai/Yi-1.5-9B-Chat,ibm-granite/granite-3.1-8b-instruct,tiiuae/Falcon3-7B-Instruct,microsoft/Phi-4-mini-instruct'

"$PYTHON" analysis/compare_full_llm_panel.py
