# iFCM expert panel — start here

The GitHub snapshot contains source, saved upstream inputs, final predictions,
metrics, figures, and the original LIDC crop backups. It deliberately excludes
Python environments and approximately 5.7 GB of training checkpoints. The
checkpoint hashes and task manifests remain in the compact results, but
full checkpoint-level verification requires the locally retained files.
ViT-Tiny/Small results and reproduction commands are in
[docs/vit_lidc_experiment.md](docs/vit_lidc_experiment.md).

This is a research codebase for Derm7pt and LIDC-IDRI. The current, leakage-aware comparison is **A** (weighted sum of predicted findings and fixed LLM edges), **B1** (intuitionistic propagation with uncertainty in image findings), and **B2** (B1 plus uncertainty in LLM edges). The measured results, denominators, and caveats are in [RESULTS.md](RESULTS.md). Only the corrected `evaluation_v2/` results are active.

## Authoritative files

- `fresh_runs/ifs_comparison/isbi-lidc-strict-v3/cnn/`: patient-separated LIDC CNN predictions and provenance.
- `fresh_runs/ifs_comparison/isbi-b1-lidc-v1/` and `isbi-b1-derm-monet-v1/`: frozen A/B1/B2 predictions, `evaluation_v2/` metrics, `supplementary_v1/` ablations, and `paper_v4/` figures/tables.
- `results/`, `results_lidc/`, `meta/`: saved upstream votes, image predictions, labels, and historical outputs. **Do not delete**; these are needed to reconstruct the results.
- `ifcm/`, `analysis/`, `kaggle/`, `scripts/`, `tests/`, `pyproject.toml`, `requirements-repro.txt`: source and environment definition. The mathematical B1/B2 specification is `docs/intuitionistic_reasoning_spec.md`.
- Obsolete pilot reports, preliminary runs, and superseded drafts were removed from this working directory. They are not inputs to the active comparison.

## Verify saved results (no training)

From this directory, use the existing environment or install a fresh one with `python3 -m venv ../.venv-repro` and `../.venv-repro/bin/python -m pip install -e . -r requirements-repro.txt`. Then:

```bash
P=../.venv-repro/bin/python
for entry in 'lidc isbi-b1-lidc-v1' 'derm isbi-b1-derm-monet-v1'; do
  set -- $entry
  for module in b1 b2 comparison supplementary paper_outputs; do
    "$P" -m "ifcm.$module" --dataset "$1" --run-id "$2" --verify
  done
  "$P" -m ifcm.comparison --dataset "$1" --run-id "$2" --parity
  "$P" -m ifcm.evaluate_comparison --dataset "$1" --run-id "$2" --summary
done
"$P" -m ifcm.safe_cnn verify-boundary --run-id isbi-lidc-strict-v3 --dry-run
.venv-upstream/bin/python -m unittest discover -s tests
```

The full test suite needs PyTorch/torchvision from `.venv-upstream` (38 tests passed here, one skipped). The smaller `../.venv-repro` environment verifies and recomputes the saved downstream results but does not include those GPU dependencies.

The primary comparison numbers are plain CSVs at `fresh_runs/ifs_comparison/<run-id>/evaluation_v2/metrics.csv`, `paired_comparisons.csv`, and `attribution.csv`; supplement files are in `supplementary_v1/`. Each run also has a **post-hoc exploratory** `source_uncertainty_v1/` analysis, documented separately in `RESULTS.md`. Paper-ready figures and tables are in `paper_v4/`. Saved predictions and provenance manifests live alongside them. The paper bundle is a draft, not a clinically reviewed final submission.

## Recompute downstream numbers into new folders

The following uses the **saved** LLM votes, Derm embeddings, and strict LIDC CNN predictions; it does not rerun the ten LLMs or train the CNN. It writes new run IDs, leaving verified results untouched. The evaluation performs 5,000 bootstrap resamples per dataset and may take a while.

```bash
P=../.venv-repro/bin/python
for entry in 'lidc reproduction-lidc-v1' 'derm reproduction-derm-v1'; do
  set -- $entry
  "$P" -m ifcm.b1 --dataset "$1" --run-id "$2" --fit
  "$P" -m ifcm.b1 --dataset "$1" --run-id "$2" --predict
  "$P" -m ifcm.b2 --dataset "$1" --run-id "$2" --fit
  "$P" -m ifcm.b2 --dataset "$1" --run-id "$2" --predict
  "$P" -m ifcm.comparison --dataset "$1" --run-id "$2" --fit
  "$P" -m ifcm.comparison --dataset "$1" --run-id "$2" --predict
  "$P" -m ifcm.comparison --dataset "$1" --run-id "$2" --parity
  "$P" -m ifcm.evaluate_comparison --dataset "$1" --run-id "$2" --evaluate
  "$P" -m ifcm.supplementary --dataset "$1" --run-id "$2" --build
  "$P" -m ifcm.paper_outputs --dataset "$1" --run-id "$2" --build
done
```

New results appear under `fresh_runs/ifs_comparison/reproduction-{lidc,derm}-v1/`. Pick different run IDs if these already exist: the fit steps intentionally refuse to overwrite an existing run. Do not use `ifcm all` in this working directory: that historical command regenerates the discarded pilot output and writes into `results/`, `results_lidc/`, and `report/`.

## Reproduction boundary

This is a **downstream reproduction from saved upstream artifacts**, not a complete raw-image/ten-LLM rerun. The strict LIDC CNN was trained with patient separation and has 60 recorded tasks; its source, manifests, and predictions are preserved. Its crop files and ImageNet checkpoint may depend on machine-local `/var/tmp`, so a fresh GPU retrain needs those inputs restored first. Only two of the ten LLMs were independently spot-checked; all ten saved panel responses are retained as research inputs. Model revisions are in `fresh_runs/models_manifest.json`, but local LLM weights under `/var/tmp` are temporary. The historical analysis code remains for audit, but its pilot report and generated output files are no longer part of the active working directory.

Some saved manifests contain absolute paths to this workspace. If the project is moved to another machine or directory, their verification commands may need path-aware restoration; a portable, clean-room reproduction has not been demonstrated.
