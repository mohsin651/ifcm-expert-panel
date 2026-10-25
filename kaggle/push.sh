#!/bin/bash
# usage: ./push.sh <job dir> <smoke|full> [part]
# IFCM_BUILD_ONLY=1 creates the run.py and metadata without contacting Kaggle.
set -euo pipefail
cd "$(dirname "$0")"
JOB=${1:?job required}; MODE=${2:?smoke or full required}; PART=${3:-}
case "$JOB" in llm-panel|llm-panel2|llm-panel3|vlm-embed|llm-lidc|lidc-prep|lidc-embed|lidc-cnn|lidc-cnn2|lidc-tcia) ;; *) echo "Unknown job: $JOB" >&2; exit 2 ;; esac
case "$MODE" in smoke|full) ;; *) echo "Mode must be smoke or full" >&2; exit 2 ;; esac
if [ -n "$PART" ]; then
  case "$PART" in 0|1|2|3) ;; *) echo "Part must be 0, 1, 2, or 3" >&2; exit 2 ;; esac
fi
KAGGLE_USER=${KAGGLE_USER:-mohsinimam}
SLUG=ifcm-$JOB-$MODE${PART:+-$PART}
D=build/$SLUG
mkdir -p "$D"
CONC=concepts.py; [ "$JOB" = lidc-prep ] && CONC=/dev/null; [ "$JOB" = lidc-embed ] && CONC=/dev/null; { [ "$JOB" = lidc-cnn ] || [ "$JOB" = lidc-cnn2 ]; } && CONC=/dev/null; [ "$JOB" = llm-lidc ] && CONC=concepts_lidc.py; [ "$JOB" = lidc-tcia ] && CONC=lidc-tcia/missing.py
BODY=$JOB/body.py; [ "$JOB" = lidc-cnn2 ] && BODY=lidc-cnn/body.py
{ echo "import os"; [ "$MODE" = smoke ] && echo 'os.environ["SMOKE"] = "1"'; [ -n "$PART" ] && echo "os.environ[\"PART\"] = \"$PART\""; [ "$JOB" = lidc-cnn2 ] && echo 'os.environ["CNN_PROFILE"] = "resnet34_5slice"'; cat "$CONC"; cat "$BODY"; } > "$D/run.py"
DS='[]'; [ "$JOB" = vlm-embed ] && DS='["menakamohanakumar/derm7pt"]'; [ "$JOB" = lidc-prep ] && DS='["washingtongold/lidcidri30"]'
KS='[]'; { [ "$JOB" = lidc-embed ] || [ "$JOB" = lidc-cnn ] || [ "$JOB" = lidc-cnn2 ]; } && KS=${KS_OVERRIDE:-"[\"$KAGGLE_USER/ifcm-lidc-prep-full-0\", \"$KAGGLE_USER/ifcm-lidc-prep-full-1\", \"$KAGGLE_USER/ifcm-lidc-prep-full-2\", \"$KAGGLE_USER/ifcm-lidc-prep-full-3\"]"}
if { [ "$JOB" = lidc-cnn ] || [ "$JOB" = lidc-cnn2 ]; } && [ "$MODE" = full ] && [ -n "$PART" ] && [ -z "${KS_OVERRIDE:-}" ]; then
  KS="[\"$KAGGLE_USER/ifcm-lidc-prep-full-0\", \"$KAGGLE_USER/ifcm-lidc-prep-full-1\", \"$KAGGLE_USER/ifcm-lidc-prep-full-2\", \"$KAGGLE_USER/ifcm-lidc-prep-full-3\", \"$KAGGLE_USER/ifcm-lidc-tcia-full-0\", \"$KAGGLE_USER/ifcm-lidc-tcia-full-1\", \"$KAGGLE_USER/ifcm-lidc-tcia-full-2\", \"$KAGGLE_USER/ifcm-lidc-tcia-full-3\"]"
fi
GPU='"enable_gpu": true, "machine_shape": "NvidiaTeslaT4"'; { [ "$JOB" = lidc-prep ] || [ "$JOB" = lidc-tcia ]; } && GPU='"enable_gpu": false'
cat > $D/kernel-metadata.json <<EOF
{"id": "$KAGGLE_USER/$SLUG", "title": "$SLUG", "code_file": "run.py", "language": "python", "kernel_type": "script",
 "is_private": true, $GPU, "enable_internet": true,
 "dataset_sources": $DS, "kernel_sources": $KS, "competition_sources": []}
EOF
python3 -I -c "import ast,sys; ast.parse(open('$D/run.py').read())"
if [ "${IFCM_BUILD_ONLY:-0}" != 1 ]; then
  kaggle kernels push -p "$D"
fi
