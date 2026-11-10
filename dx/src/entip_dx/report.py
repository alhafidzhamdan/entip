"""Draft layered report text (WHO CNS5 Table 4 layout: integrated diagnosis, histological diagnosis,
CNS WHO grade, molecular information). Every AI line says so and gives its confidence; anything without a
result reads "pending". Age, site and imaging are never written here."""
from __future__ import annotations

from .contract import TESTS  # noqa: F401  (test names appear verbatim)

LABELS = {
    "adequacy": "Adequacy", "tissue_class": "Tissue", "glioma_pattern": "Growth pattern",
    "necrosis": "Necrosis", "microvascular_proliferation": "Microvascular proliferation",
    "mitotic_activity": "Mitotic activity", "IDH": "IDH", "ATRX": "ATRX", "TP53": "TP53",
    "1p19q": "1p/19q", "CDKN2A": "CDKN2A/B", "EGFR": "EGFR", "chr7_chr10": "Chromosomes 7/10",
    "TERT": "TERT promoter", "H3K27": "H3 K27", "H3G34": "H3 G34", "MGMT": "MGMT promoter",
    "methylation_class": "Methylation class",
}

WORDS = {
    "present": "present", "absent": "absent", "mutant": "mutant", "wildtype": "wildtype",
    "retained": "retained", "lost": "loss of nuclear expression", "codeleted": "codeleted",
    "not_codeleted": "not codeleted", "homozygous_deletion": "homozygous deletion",
    "amplified": "amplified", "not_amplified": "not amplified", "gain7_loss10": "+7/-10 present",
    "absent_cn": "no +7/-10", "altered": "altered", "methylated": "methylated", "unmethylated": "unmethylated",
    "glioma": "glioma", "normal_brain": "normal brain", "reactive": "reactive", "metastasis": "metastasis",
    "lymphoma": "lymphoma", "other_primary_cns": "other primary CNS tumour", "adequate": "adequate",
    "inadequate": "inadequate", "diffuse_astrocytic": "diffuse astrocytic", "oligodendroglial": "oligodendroglial",
}

TEST_WORDS = {
    "pathologist_review": "Pathologist review", "additional_sampling": "Further tissue / levels",
    "IDH1_R132H_IHC": "IDH1 R132H immunohistochemistry", "IDH_sequencing": "IDH1/IDH2 sequencing",
    "ATRX_IHC": "ATRX immunohistochemistry", "p53_IHC": "p53 immunohistochemistry",
    "1p19q_testing": "1p/19q codeletion testing", "CDKN2A_B_testing": "CDKN2A/B deletion testing",
    "EGFR_amplification_testing": "EGFR amplification testing",
    "chr7_chr10_copy_number": "Chromosome 7/10 copy number", "TERT_promoter_sequencing":
    "TERT promoter sequencing", "H3K27M_IHC": "H3 K27M immunohistochemistry",
    "H3K27me3_IHC": "H3 K27me3 immunohistochemistry", "H3G34_testing": "H3 G34 testing",
    "MGMT_promoter_methylation": "MGMT promoter methylation", "methylation_array": "DNA methylation array",
}


def _word(finding: str, call: str | None) -> str:
    if call is None:
        return "pending"
    if finding == "chr7_chr10" and call == "absent":
        return "no +7/-10"
    return WORDS.get(call, call.replace("_", " "))


def _line(row: dict) -> str:
    label = LABELS.get(row["finding"], row["finding"])
    if row["status"] == "pending":
        guess = _word(row["finding"], row["call"])
        conf = row.get("confidence")
        tail = f" (AI suggests {guess}, {conf:.2f}, below threshold)" if conf is not None else ""
        return f"- {label}: pending{tail}"
    word = _word(row["finding"], row["call"])
    if row["finding"] == "methylation_class" and row["value"].get("class"):
        word = f"{word} ({row['value']['class']})"
    if row["source"] == "ai":
        return f"- {label}: {word} [AI {row['model_id']}, confidence {row['confidence']:.2f}]"
    how = row.get("method") or ("pathologist" if row["source"] == "pathologist" else "test")
    return f"- {label}: {word} [{how}]"


def render(dx: dict) -> str:
    d = dx["integrated_diagnosis"]
    grade = d["cns_who_grade"]
    hist_rows = dx["layers"]["histology"] + dx["layers"]["grade_features"]
    lines = [
        "DRAFT: AI-assisted integrated report for pathologist review. Research use only; not for clinical use.",
        "",
        f"Integrated diagnosis: {d['text']}",
    ]
    if d.get("favoured") and d["status"] != "complete":
        lines.append(f"Favoured while pending: {d['favoured']}")
    lines += [
        f"CNS WHO grade: {grade if grade is not None else 'pending'}",
        "",
        "Histological findings:",
        *([_line(r) for r in hist_rows] or ["- pending"]),
        "",
        "Molecular information:",
        *([_line(r) for r in dx["layers"]["molecular"]] or ["- pending"]),
    ]
    if dx["layers"]["methylation"]:
        lines += ["", "DNA methylation:", *[_line(r) for r in dx["layers"]["methylation"]]]
    if dx["based_on"]:
        lines += ["", "Based on (WHO CNS5 criteria):"]
        for b in dx["based_on"]:
            mark = {True: "met", False: "not met", None: "pending"}[b["met"]]
            lines.append(f"- {b['criterion']}: {mark}")
    if dx["orders"]:
        lines += ["", "Tests requested before sign-out:" if any(o["blocking"] for o in dx["orders"])
                  else "Tests requested:"]
        for o in dx["orders"]:
            flag = "" if o["blocking"] else " (does not change the diagnosis)"
            lines.append(f"- {TEST_WORDS.get(o['test'], o['test'])}: {o['reason']}{flag}")
    if dx["conflicts"]:
        lines += ["", "Discrepancies:", *[f"- {x}" for x in dx["conflicts"]]]
    if dx["notes"]:
        lines += ["", "Comment:", *[f"- {x}" for x in dx["notes"]]]
    refs = sorted({b["citation"] for b in dx["based_on"]})
    if refs:
        lines += ["", "References:", *[f"- {r}" for r in refs]]
    lines += ["", f"Engine {dx['engine']['name']} {dx['engine']['version']}, rules {dx['engine']['rules']}. "
                  "Awaiting pathologist sign-off."]
    return "\n".join(lines)
