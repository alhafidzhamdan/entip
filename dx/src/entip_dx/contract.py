"""The shared output contract for the AI glioma workflow (steps 1 to 4) and the engine's output (step 5).

Every step emits one *finding record* per finding: a triage call, one high-grade feature, one panel marker,
or a methylation class. Real test results and pathologist calls use the same record with ``source`` set to
``"test"`` or ``"pathologist"``, so the engine treats every input the same way.

This module is the single source of truth. ``entip-dx export-contract`` writes ``dx/contract/*.json`` from it.
"""
from __future__ import annotations

from typing import Any

STEP_CONTRACT = "entip.glioma_step/1"
DX_CONTRACT = "entip.integrated_dx/1"

STEPS = ("triage", "high_grade_features", "glioma_panel", "methylation_class")

# Allowed calls per finding. "indeterminate" is allowed for every finding and forces status "uncertain".
FINDINGS: dict[str, dict[str, tuple[str, ...]]] = {
    "triage": {
        "adequacy": ("adequate", "inadequate"),
        "tissue_class": ("normal_brain", "reactive", "glioma", "other_primary_cns", "metastasis", "lymphoma"),
        # Optional: growth pattern / lineage on H&E. Only "diffuse_astrocytic" and "oligodendroglial" are used.
        "glioma_pattern": ("diffuse_astrocytic", "oligodendroglial", "circumscribed", "ependymal"),
    },
    "high_grade_features": {
        "necrosis": ("present", "absent"),
        "microvascular_proliferation": ("present", "absent"),
        "mitotic_activity": ("present", "absent"),  # "present" = elevated for a diffuse glioma
    },
    "glioma_panel": {
        # Marker names match db genomic_finding.marker.
        "IDH": ("mutant", "wildtype"),
        "ATRX": ("retained", "lost"),
        "TP53": ("mutant", "wildtype"),
        "1p19q": ("codeleted", "not_codeleted"),
        "CDKN2A": ("homozygous_deletion", "retained"),  # CDKN2A/B
        "EGFR": ("amplified", "not_amplified"),
        "chr7_chr10": ("gain7_loss10", "absent"),
        "TERT": ("mutant", "wildtype"),  # promoter
        "H3K27": ("altered", "wildtype"),  # K27M mutation or H3 K27me3 loss / EZHIP
        "H3G34": ("mutant", "wildtype"),
        "MGMT": ("methylated", "unmethylated"),  # promoter; prognostic/predictive, never changes the type
    },
    "methylation_class": {
        # call = family, mapped to WHO CNS5 types; the classifier's own subclass goes in value.class
        "methylation_class": (
            "glioblastoma_idh_wildtype",
            "astrocytoma_idh_mutant",
            "oligodendroglioma_idh_mutant_codeleted",
            "diffuse_midline_glioma_h3k27",
            "diffuse_hemispheric_glioma_h3g34",
            "pediatric_hgg_h3wt_idhwt",
            "control_tissue",
            "other",
            "no_match",
        ),
    },
}

INDETERMINATE = "indeterminate"
SOURCES = ("ai", "test", "pathologist")
STATUSES = ("confident", "uncertain", "not_assessed")

# Tests a step (or the engine) may order. Keep names stable: the GUI and the db show them.
TESTS = (
    "pathologist_review",
    "additional_sampling",
    "IDH1_R132H_IHC",
    "IDH_sequencing",
    "ATRX_IHC",
    "p53_IHC",
    "1p19q_testing",
    "CDKN2A_B_testing",
    "EGFR_amplification_testing",
    "chr7_chr10_copy_number",
    "TERT_promoter_sequencing",
    "H3K27M_IHC",
    "H3K27me3_IHC",
    "H3G34_testing",
    "MGMT_promoter_methylation",
    "methylation_array",
)

# The real test the engine orders when a finding is uncertain or missing.
DEFAULT_TEST: dict[str, str] = {
    "adequacy": "additional_sampling",
    "tissue_class": "pathologist_review",
    "glioma_pattern": "pathologist_review",
    "necrosis": "pathologist_review",
    "microvascular_proliferation": "pathologist_review",
    "mitotic_activity": "pathologist_review",
    "IDH": "IDH1_R132H_IHC",
    "ATRX": "ATRX_IHC",
    "TP53": "p53_IHC",
    "1p19q": "1p19q_testing",
    "CDKN2A": "CDKN2A_B_testing",
    "EGFR": "EGFR_amplification_testing",
    "chr7_chr10": "chr7_chr10_copy_number",
    "TERT": "TERT_promoter_sequencing",
    "H3K27": "H3K27M_IHC",
    "H3G34": "H3G34_testing",
    "MGMT": "MGMT_promoter_methylation",
    "methylation_class": "methylation_array",
}

# Keys that must never appear in a finding record or in the engine output. Age, site and radiology are a
# prior only (passed separately to the engine) and are never written into a report.
FORBIDDEN_KEYS = ("age", "age_years", "site", "location", "radiology", "imaging", "sex",
                  "chi", "nhs_number", "dob", "patient_name", "accession", "postcode")

EVIDENCE_KINDS = ("scan", "region", "cell", "tile", "prediction", "annotation", "genomic_finding", "assay")


def step_of(finding: str) -> str:
    for step, names in FINDINGS.items():
        if finding in names:
            return step
    raise KeyError(finding)


class ContractError(ValueError):
    pass


def _walk_keys(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}" if path else k, k
            yield from _walk_keys(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def check_no_priors(obj: Any, where: str) -> None:
    for path, key in _walk_keys(obj):
        if key.lower() in FORBIDDEN_KEYS:
            raise ContractError(f"{where}: key '{path}' is not allowed (priors and identifiers stay out)")


def validate_finding(rec: dict) -> dict:
    """Check one finding record and return it with defaults filled in. Raises ContractError."""
    if not isinstance(rec, dict):
        raise ContractError("a finding record must be an object")
    r = dict(rec)
    if r.get("contract", STEP_CONTRACT) != STEP_CONTRACT:
        raise ContractError(f"contract must be '{STEP_CONTRACT}', got {r.get('contract')!r}")
    r["contract"] = STEP_CONTRACT
    for key in ("step", "finding", "study_id", "call"):
        if not r.get(key):
            raise ContractError(f"missing '{key}'")
    step, finding = r["step"], r["finding"]
    if step not in FINDINGS:
        raise ContractError(f"unknown step {step!r}; one of {STEPS}")
    if finding not in FINDINGS[step]:
        raise ContractError(f"unknown finding {finding!r} for step {step!r}; one of {tuple(FINDINGS[step])}")
    allowed = FINDINGS[step][finding] + (INDETERMINATE,)
    if r["call"] not in allowed:
        raise ContractError(f"{finding}: call {r['call']!r} not in {allowed}")
    r.setdefault("source", "ai")
    if r["source"] not in SOURCES:
        raise ContractError(f"source must be one of {SOURCES}")
    r.setdefault("scan_ids", [])
    if r["source"] == "ai":
        conf = r.get("confidence")
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            raise ContractError(f"{finding}: an AI record needs confidence in [0, 1]")
        thr = r.get("threshold")
        if thr is not None and (not isinstance(thr, (int, float)) or not 0.0 <= thr <= 1.0):
            raise ContractError(f"{finding}: threshold must be in [0, 1]")
        if not r.get("model") or not r["model"].get("model_id"):
            raise ContractError(f"{finding}: an AI record needs model.model_id")
        if "status" not in r:
            r["status"] = "confident" if (thr is not None and conf >= thr) else "uncertain"
    else:
        r.setdefault("confidence", 1.0)
        r.setdefault("status", "confident")
    if r["status"] not in STATUSES:
        raise ContractError(f"status must be one of {STATUSES}")
    if r["call"] == INDETERMINATE and r["status"] == "confident":
        raise ContractError(f"{finding}: an indeterminate call cannot be confident")
    if r.get("order_test") is not None and r["order_test"] not in TESTS:
        raise ContractError(f"order_test {r['order_test']!r} not in the test list")
    if r["source"] == "ai" and r["status"] == "uncertain" and not r.get("order_test"):
        r["order_test"] = DEFAULT_TEST[finding]
    for ev in r.get("evidence", []) or []:
        if not isinstance(ev, dict) or ev.get("kind") not in EVIDENCE_KINDS or not ev.get("id"):
            raise ContractError(f"{finding}: evidence items need kind (one of {EVIDENCE_KINDS}) and id")
    prov = r.get("provenance")
    if prov is None:
        if r["source"] == "ai":
            prov = {"kind": "model", "actor": r["model"]["model_id"], "version": r["model"].get("version")}
        else:
            raise ContractError(f"{finding}: test and pathologist records need provenance")
    if prov.get("kind") not in ("person", "model", "program") or not prov.get("actor"):
        raise ContractError(f"{finding}: provenance needs kind (person|model|program) and actor")
    r["provenance"] = prov
    check_no_priors(r, f"finding {finding}")
    return r


def step_json_schema() -> dict:
    """JSON Schema (draft 2020-12) for one finding record."""
    one_of = []
    for step, names in FINDINGS.items():
        for finding, calls in names.items():
            one_of.append({
                "properties": {"step": {"const": step}, "finding": {"const": finding},
                               "call": {"enum": list(calls) + [INDETERMINATE]}},
            })
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/alhafidzhamdan/entip/dx/contract/glioma_step.schema.json",
        "title": STEP_CONTRACT,
        "description": "One finding from one step of the AI glioma workflow (or a real test / pathologist call).",
        "type": "object",
        "required": ["contract", "step", "finding", "study_id", "call"],
        "properties": {
            "contract": {"const": STEP_CONTRACT},
            "step": {"enum": list(STEPS)},
            "finding": {"type": "string"},
            "study_id": {"type": "string"},
            "scan_ids": {"type": "array", "items": {"type": "string"}},
            "call": {"type": "string"},
            "source": {"enum": list(SOURCES), "default": "ai"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                           "description": "Calibrated probability that the call is right."},
            "threshold": {"type": "number", "minimum": 0, "maximum": 1,
                          "description": "Calibrated accept threshold for this finding and model version."},
            "status": {"enum": list(STATUSES),
                       "description": "confident = use it; uncertain = order the real test."},
            "probabilities": {"type": "object", "additionalProperties": {"type": "number"}},
            "value": {"type": "object",
                      "description": "Extras, e.g. mitoses_per_mm2, area_mm2; methylation: class, classifier."},
            "evidence": {"type": "array", "items": {
                "type": "object", "required": ["kind", "id"],
                "properties": {"kind": {"enum": list(EVIDENCE_KINDS)}, "id": {"type": "string"},
                               "x_um": {"type": "number"}, "y_um": {"type": "number"},
                               "w_um": {"type": "number"}, "h_um": {"type": "number"},
                               "weight": {"type": "number"}}}},
            "order_test": {"anyOf": [{"type": "null"}, {"enum": list(TESTS)}]},
            "reasons": {"type": "array", "items": {"type": "string"}},
            "model": {"type": "object", "required": ["model_id"],
                      "properties": {"model_id": {"type": "string"}, "version": {"type": "string"},
                                     "licence": {"type": "string"}}},
            "provenance": {"type": "object", "required": ["kind", "actor"],
                           "properties": {"kind": {"enum": ["person", "model", "program"]},
                                          "actor": {"type": "string"}, "version": {"type": "string"}}},
        },
        "allOf": [{"oneOf": one_of}],
        "not": {"anyOf": [{"required": [k]} for k in FORBIDDEN_KEYS]},
    }


# Real results arrive from db genomic_finding in lab words. Map them onto contract calls.
MARKER_ALIASES = {
    "IDH": "IDH", "IDH1": "IDH", "IDH2": "IDH", "ATRX": "ATRX", "TP53": "TP53", "p53": "TP53",
    "1p19q": "1p19q", "1p/19q": "1p19q", "CDKN2A": "CDKN2A", "CDKN2A_B": "CDKN2A", "CDKN2A/B": "CDKN2A",
    "EGFR": "EGFR", "EGFR_amp": "EGFR", "chr7_chr10": "chr7_chr10", "+7/-10": "chr7_chr10",
    "TERT": "TERT", "TERT_promoter": "TERT", "H3K27": "H3K27", "H3K27M": "H3K27", "H3_K27": "H3K27",
    "H3K27me3": "H3K27", "H3G34": "H3G34", "H3_G34": "H3G34", "MGMT": "MGMT", "MGMT_promoter": "MGMT",
}

RESULT_ALIASES = {
    "mutant": "mutant", "mutated": "mutant", "mutation": "mutant", "positive": None,
    "wildtype": "wildtype", "wild-type": "wildtype", "wild type": "wildtype", "wt": "wildtype",
    "retained": "retained", "loss": "lost", "lost": "lost",
    "codeleted": "codeleted", "co-deleted": "codeleted", "not codeleted": "not_codeleted",
    "not_codeleted": "not_codeleted", "intact": None,
    "homozygous deletion": "homozygous_deletion", "homozygous_deletion": "homozygous_deletion",
    "no deletion": "retained",
    "amplified": "amplified", "not amplified": "not_amplified", "not_amplified": "not_amplified",
    "+7/-10": "gain7_loss10", "gain7_loss10": "gain7_loss10", "no +7/-10": "absent", "absent": "absent",
    "altered": "altered", "methylated": "methylated", "unmethylated": "unmethylated",
}

# Words whose meaning depends on the marker.
_MARKER_RESULT = {
    ("CDKN2A", "intact"): "retained", ("1p19q", "intact"): "not_codeleted",
    ("H3K27", "mutant"): "altered", ("H3K27", "positive"): "altered", ("H3K27", "lost"): "altered",
    ("H3K27", "retained"): "wildtype", ("TP53", "positive"): "mutant", ("IDH", "positive"): "mutant",
    ("TP53", "aberrant"): "mutant",
}


def normalise_result(marker: str, result: str) -> tuple[str, str] | None:
    """Map a db marker and lab result to (finding, call), or None when it is not a panel marker we use."""
    m = MARKER_ALIASES.get(marker) or MARKER_ALIASES.get(marker.strip())
    if m is None:
        return None
    r = result.strip().lower()
    call = _MARKER_RESULT.get((m, r))
    if call is None:
        call = RESULT_ALIASES.get(r)
    if call is None or call not in FINDINGS["glioma_panel"][m]:
        return None
    return m, call


DX_STATUSES = ("complete", "provisional", "pending", "not_diffuse_glioma", "inadequate")
LAYER_SOURCES = SOURCES + ("rule", "missing")


def dx_json_schema() -> dict:
    """JSON Schema for the engine's output, one per patient."""
    layer_item = {
        "type": "object",
        "required": ["finding", "call", "source", "status"],
        "properties": {
            "finding": {"type": "string"}, "call": {"type": ["string", "null"]},
            "source": {"enum": list(LAYER_SOURCES)}, "status": {"enum": list(STATUSES) + ["pending"]},
            "confidence": {"type": ["number", "null"]}, "model_id": {"type": ["string", "null"]},
            "evidence": {"type": "array"},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/alhafidzhamdan/entip/dx/contract/integrated_dx.schema.json",
        "title": DX_CONTRACT,
        "description": "WHO CNS5 integrated diagnosis draft for pathologist sign-off. Research use only.",
        "type": "object",
        "required": ["contract", "study_id", "engine", "integrated_diagnosis", "layers", "based_on",
                     "orders", "conflicts", "report_text", "signoff"],
        "properties": {
            "contract": {"const": DX_CONTRACT},
            "study_id": {"type": "string"},
            "engine": {"type": "object", "required": ["name", "version", "rules"],
                       "properties": {"name": {"type": "string"}, "version": {"type": "string"},
                                      "rules": {"type": "string"}}},
            "integrated_diagnosis": {
                "type": "object", "required": ["text", "status"],
                "properties": {
                    "text": {"type": "string"},
                    "who_type": {"type": ["string", "null"]},
                    "cns_who_grade": {"type": ["integer", "string", "null"],
                                      "description": "2, 3, 4, or '2 or 3' when histology must decide"},
                    "suffix": {"enum": [None, "NOS", "NEC"]},
                    "status": {"enum": list(DX_STATUSES)},
                    "favoured": {"type": ["string", "null"],
                                 "description": "the most likely type while tests are pending"},
                },
            },
            "layers": {
                "type": "object",
                "properties": {k: {"type": "array", "items": layer_item}
                               for k in ("histology", "grade_features", "molecular", "methylation")},
            },
            "based_on": {"type": "array", "items": {
                "type": "object", "required": ["criterion", "met", "citation"],
                "properties": {"criterion": {"type": "string"}, "met": {"type": ["boolean", "null"]},
                               "findings": {"type": "array", "items": {"type": "string"}},
                               "citation": {"type": "string"}}}},
            "orders": {"type": "array", "items": {
                "type": "object", "required": ["test", "reason"],
                "properties": {"test": {"enum": list(TESTS)}, "reason": {"type": "string"},
                               "finding": {"type": ["string", "null"]},
                               "blocking": {"type": "boolean",
                                            "description": "true if the diagnosis cannot be finalised without it"}}}},
            "conflicts": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "array", "items": {"type": "string"}},
            "ai_decisive": {"type": "array", "items": {"type": "string"},
                            "description": "findings that met a criterion on AI calls alone"},
            "ai_disagreements": {"type": "array", "items": {"type": "string"},
                                 "description": "confident AI calls overruled by a test or pathologist"},
            "report_text": {"type": "string"},
            "signoff": {"type": "object", "required": ["required", "status"],
                        "properties": {"required": {"const": True},
                                       "status": {"enum": ["awaiting_pathologist", "signed", "amended"]}}},
        },
        "not": {"anyOf": [{"required": [k]} for k in FORBIDDEN_KEYS]},
    }
