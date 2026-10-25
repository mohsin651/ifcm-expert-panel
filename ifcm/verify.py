"""Check that logs, summaries, generated values, and report artifacts agree."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("Verification failed: " + message)


def summary(encoder: str) -> dict:
    path = ROOT / "results_lidc" / f"summary_{encoder}_lidc_cnn_split0.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def log_number(log: str, pattern: str) -> float:
    found = re.search(pattern, log, re.M)
    require(found is not None, f"missing log pattern {pattern}")
    return float(found.group(1))


def main() -> None:
    summaries = {}
    for encoder in ("biomedclip", "dinov3", "cnn"):
        result = summary(encoder)
        text = (ROOT / "results_lidc" / f"log_{encoder}.txt").read_text(encoding="utf-8")
        attr = result["attr: perc_bern - know_flip"]
        auc_log = log_number(text, r"^\s*perc_bern\s+- know_flip AUROC ([0-9.]+)")
        require(abs(attr - auc_log) < 0.0005, f"{encoder}: attribution differs between log and summary")
        diagnosis = result["auc: predicted findings + LLM panel edges [OURS]"]
        diagnosis_log = log_number(text, r"^\s*predicted findings \+ LLM panel edges \[OURS\] ([0-9.]+) /")
        require(abs(diagnosis - diagnosis_log) < 0.0005, f"{encoder}: diagnosis differs between log and summary")
        require(result["n_errors"] > result["n_perception_errors"] > 0,
                f"{encoder}: invalid error counts")
        paired = result["attribution_bootstrap"]["paired_difference"]
        baseline = result["attr: 1-conf"]
        require(abs(paired["point"] - (attr - baseline)) < 1e-12,
                f"{encoder}: paired difference does not equal the two AUROCs")
        require(paired["ci95"][0] <= paired["ci95"][1] and 0 <= paired["one_sided_p"] <= 1,
                f"{encoder}: invalid bootstrap result")
        summaries[encoder] = result

    cnn = summaries["cnn"]
    generated = (ROOT / "report" / "generated_metrics.tex").read_text(encoding="utf-8")
    expected = {
        "CNNAttrAUC": f"{cnn['attr: perc_bern - know_flip']:.2f}",
        "CNNUncAUC": f"{cnn['attr: 1-conf']:.2f}",
        "CNNAttrGain": f"{cnn['attribution_bootstrap']['paired_difference']['point']:+.2f}",
        "CNNAttrTail": f"{cnn['attribution_bootstrap']['paired_difference']['one_sided_p']:.3f}",
    }
    for name, value in expected.items():
        require(f"\\newcommand{{\\{name}}}{{{value}}}" in generated,
                f"report macro {name} is stale")

    report = ROOT / "report" / "report.pdf"
    figures = [ROOT / "report" / "figs" / f"{name}.pdf" for name in
               ("derm_edges", "lidc_edges", "diagnosis", "attribution", "cnn_blur", "lidc_examples")]
    for artifact in [report, *figures]:
        require(artifact.is_file() and artifact.read_bytes()[:4] == b"%PDF",
                f"missing or invalid PDF: {artifact}")
    sources = [ROOT / "report" / name for name in
               ("report.tex", "generated_metrics.tex", "generated_readers_rows.tex")]
    require(report.stat().st_mtime >= max(source.stat().st_mtime for source in [*sources, *figures]),
            "report PDF is older than its source or figures; run ifcm report")
    print("Verified: LIDC logs and summaries agree; report values and seven PDFs are current.")
    print(f"CNN attribution {expected['CNNAttrAUC']}; plain uncertainty {expected['CNNUncAUC']}; "
          f"paired gain {expected['CNNAttrGain']}; one-sided bootstrap tail fraction {expected['CNNAttrTail']}.")


if __name__ == "__main__":
    main()
