"""Write gui/web/mock/drafts.json: WHO CNS5 draft reports for the mock patients, made by the dx engine.

Each mock patient's molecular fields become real-test records; histology comes from made-up AI records
(necrosis and microvascular proliferation by diagnosis). Synthetic data only. Needs `pip install -e dx`.
"""
from __future__ import annotations

import json
from pathlib import Path

from entip_dx.engine import diagnose

HERE = Path(__file__).resolve().parents[1] / "web" / "mock"
MOCK_MODEL = {"model_id": "mock-head", "version": "0.0.0", "licence": "MIT"}

# mock field -> (contract finding, {mock value -> call})
FIELDS = {
    "IDH": ("IDH", lambda v: "wildtype" if v == "wildtype" else "mutant"),
    "1p/19q": ("1p19q", lambda v: {"codeleted": "codeleted", "intact": "not_codeleted"}.get(v)),
    "ATRX": ("ATRX", lambda v: {"loss": "lost", "retained": "retained"}.get(v)),
    "TP53": ("TP53", lambda v: v if v in ("mutant", "wildtype") else None),
    "CDKN2A/B": ("CDKN2A", lambda v: {"homozygous deletion": "homozygous_deletion", "retained": "retained"}.get(v)),
    "EGFR": ("EGFR", lambda v: {"amplified": "amplified", "not amplified": "not_amplified"}.get(v)),
    "+7/-10": ("chr7_chr10", lambda v: {"present": "gain7_loss10", "absent": "absent"}.get(v)),
    "TERT promoter": ("TERT", lambda v: "wildtype" if v == "wildtype" else "mutant"),
    "MGMT": ("MGMT", lambda v: v if v in ("methylated", "unmethylated") else None),
}


def records(p: dict) -> list[dict]:
    sid = p["study_id"]
    ai = lambda step, f, call, c=0.95: {"step": step, "finding": f, "study_id": sid, "call": call,  # noqa: E731
                                        "confidence": c, "threshold": 0.9, "model": MOCK_MODEL}
    out = [ai("triage", "adequacy", "adequate"), ai("triage", "tissue_class", "glioma")]
    gbm = "Glioblastoma" in (p.get("diagnosis") or "")
    out += [ai("high_grade_features", "necrosis", "present" if gbm else "absent"),
            ai("high_grade_features", "microvascular_proliferation", "present" if gbm else "absent", 0.82),
            ai("high_grade_features", "mitotic_activity", "present" if gbm else "absent")]
    # H3 by AI with high confidence for the mock (adult hemispheric cases)
    out += [ai("glioma_panel", "H3K27", "wildtype", 0.97), ai("glioma_panel", "H3G34", "wildtype", 0.96)]
    for k, v in (p.get("molecular") or {}).items():
        if k not in FIELDS:
            continue
        finding, fn = FIELDS[k]
        call = fn(v)
        if call:
            out.append({"step": "glioma_panel", "finding": finding, "study_id": sid, "call": call,
                        "source": "test", "value": {"method": "mock lab"},
                        "provenance": {"kind": "program", "actor": "mock-lab"}})
    return out


def main() -> None:
    data = json.loads((HERE / "patients.json").read_text())
    drafts = {p["study_id"]: diagnose(records(p)) for p in data["patients"]}
    (HERE / "drafts.json").write_text(json.dumps(drafts, indent=1) + "\n")
    print(f"wrote {len(drafts)} drafts to {HERE / 'drafts.json'}")


if __name__ == "__main__":
    main()
