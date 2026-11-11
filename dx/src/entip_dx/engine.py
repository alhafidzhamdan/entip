"""Step 5: combine finding records into a WHO CNS5 integrated diagnosis draft.

``diagnose(records, priors=None)`` takes finding records (``entip.glioma_step/1``) from the AI steps and from
real tests, and returns one ``entip.integrated_dx/1`` dict. It never learns anything: the decisions are the
written rules in ``rules.py``. Anything uncertain or missing that would change the diagnosis is ordered as a
real test, and the diagnosis stays pending until it comes back.

Priors (age, site, radiology) are accepted for future use (for example to weight a step's confidence) but no
rule uses them yet, and they never reach the output: the output is checked for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import __version__
from .contract import DEFAULT_TEST, DX_CONTRACT, INDETERMINATE, check_no_priors, validate_finding
from .rules import CRITERIA, FAMILY_TO_TYPE, RULES_VERSION, TYPES, cite

SOURCE_RANK = {"test": 3, "pathologist": 2, "ai": 1}

LABELS = {
    "adequacy": "Adequacy", "tissue_class": "Tissue class", "glioma_pattern": "Growth pattern",
    "necrosis": "Necrosis", "microvascular_proliferation": "Microvascular proliferation",
    "mitotic_activity": "Mitotic activity", "IDH": "IDH", "ATRX": "ATRX", "TP53": "TP53",
    "1p19q": "1p/19q", "CDKN2A": "CDKN2A/B", "EGFR": "EGFR", "chr7_chr10": "Chromosome 7/10",
    "TERT": "TERT promoter", "H3K27": "H3 K27", "H3G34": "H3 G34", "MGMT": "MGMT promoter",
    "methylation_class": "Methylation class",
}

CALL_WORDS = {
    "gain7_loss10": "+7/-10 present", "not_codeleted": "not codeleted",
    "homozygous_deletion": "homozygous deletion", "not_amplified": "not amplified",
    "normal_brain": "normal brain", "other_primary_cns": "other primary CNS tumour",
    "diffuse_astrocytic": "diffuse astrocytic", "lost": "loss of expression",
}


@dataclass
class Resolved:
    finding: str
    rec: dict | None = None          # the record used
    uncertain: dict | None = None    # best uncertain AI record, if no usable one

    @property
    def call(self) -> str | None:
        return self.rec["call"] if self.rec else None

    @property
    def source(self) -> str:
        return self.rec["source"] if self.rec else "missing"

    @property
    def guess(self) -> str | None:
        """The uncertain AI call, to show what is favoured while the test is pending."""
        u = self.uncertain
        return u["call"] if u and u["call"] != INDETERMINATE else None

    def is_(self, *calls: str) -> bool:
        return self.call in calls


@dataclass
class Ctx:
    f: dict[str, Resolved]
    priors: dict
    orders: list[dict] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    based_on: list[dict] = field(default_factory=list)
    ai_disagreements: list[str] = field(default_factory=list)

    def get(self, name: str) -> Resolved:
        return self.f.setdefault(name, Resolved(name))

    def order(self, finding: str | None, reason: str, blocking: bool = True, test: str | None = None) -> None:
        if test is None:
            u = self.get(finding).uncertain if finding else None
            test = (u or {}).get("order_test") or DEFAULT_TEST[finding]
        if any(o["test"] == test for o in self.orders):
            for o in self.orders:
                if o["test"] == test and blocking:
                    o["blocking"] = True
            return
        self.orders.append({"test": test, "reason": reason, "finding": finding, "blocking": blocking})

    def need(self, finding: str, why: str, blocking: bool = True) -> bool:
        """True if the finding has a usable call; otherwise order its test and return False."""
        r = self.get(finding)
        if r.call is not None:
            return True
        state = "uncertain" if r.uncertain else "not available"
        self.order(finding, f"{LABELS[finding]} {state}; needed to {why}", blocking)
        return False

    def basis(self, key: str, met: bool | None, *findings: str) -> None:
        text, src = CRITERIA[key]
        self.based_on.append({"criterion": text, "met": met, "findings": list(findings), "citation": cite(src)})


HISTO = ("necrosis", "microvascular_proliferation", "mitotic_activity")
NON_LESIONAL = ("normal_brain", "reactive")


def _is_ihc(rec: dict) -> bool:
    return "ihc" in str((rec.get("value") or {}).get("method") or "").lower()


def _human_beats(new: dict, old: dict | None) -> bool:
    """Test beats pathologist. Among tests, IDH sequencing beats IHC; otherwise the later record wins."""
    if old is None:
        return True
    if SOURCE_RANK[new["source"]] != SOURCE_RANK[old["source"]]:
        return SOURCE_RANK[new["source"]] > SOURCE_RANK[old["source"]]
    if new["finding"] == "IDH" and new["source"] == "test" and _is_ihc(new) != _is_ihc(old):
        return not _is_ihc(new)
    return True


def _usable(rec: dict) -> bool:
    return rec["call"] != INDETERMINATE and (rec["source"] != "ai" or rec["status"] == "confident")


def resolve(records: list[dict]) -> tuple[dict[str, Resolved], list[str], list[str]]:
    """Pick one call per finding. Returns (resolved, conflicts, ai_disagreements).

    - A real test beats a pathologist's call, which beats AI.
    - AI records are grouped by the scans they cover; the latest record per group wins, so a newer model run
      replaces an older one even when it is less sure.
    - Across scans: a high-grade feature is present if any scan shows it confidently, and absent only when every
      scan is confidently absent. A triage call of normal brain or reactive on one scan gives way to a tumour
      call on another. Other findings must agree across scans, or they are treated as uncertain.
    """
    human: dict[str, dict] = {}
    ai: dict[str, dict[tuple, dict]] = {}
    for rec in records:
        f = rec["finding"]
        if rec["source"] == "ai":
            ai.setdefault(f, {})[tuple(sorted(rec.get("scan_ids") or []))] = rec
        elif rec["call"] != INDETERMINATE and _human_beats(rec, human.get(f)):
            human[f] = rec
    out: dict[str, Resolved] = {}
    conflicts: list[str] = []
    disagree: list[str] = []
    for f in set(human) | set(ai):
        r = out[f] = Resolved(f)
        groups = list(ai.get(f, {}).values())
        sure = [g for g in groups if _usable(g)]
        unsure = [g for g in groups if not _usable(g)]
        if unsure:
            r.uncertain = max(unsure, key=lambda g: g.get("confidence") or 0)
        if f in human:
            r.rec = human[f]
            wrong = sorted({g["call"] for g in sure if g["call"] != r.rec["call"]})
            if wrong:
                disagree.append(f"{LABELS[f]}: AI said {', '.join(wrong)}; {r.rec['source']} says {r.rec['call']}")
            continue
        if not sure:
            continue
        calls = {g["call"] for g in sure}
        if f in HISTO or f == "adequacy":
            hit = "present" if f in HISTO else "adequate"
            pos = [g for g in sure if g["call"] == hit]
            if pos:
                r.rec = max(pos, key=lambda g: g.get("confidence") or 0)
            elif not unsure:
                r.rec = sure[-1]
            continue
        if f == "tissue_class" and len(calls) > 1 and len(calls - set(NON_LESIONAL)) == 1:
            tumour = (calls - set(NON_LESIONAL)).pop()
            r.rec = max((g for g in sure if g["call"] == tumour), key=lambda g: g.get("confidence") or 0)
            continue
        if len(calls) > 1:
            conflicts.append(f"{LABELS[f]}: AI calls differ between scans ({', '.join(sorted(calls))})")
            r.uncertain = r.uncertain or sure[-1]
            continue
        r.rec = sure[-1]
    return out, conflicts, disagree


def diagnose(records: list[dict], priors: dict | None = None, study_id: str | None = None) -> dict:
    recs = [validate_finding(r) for r in records]
    ids = {r["study_id"] for r in recs} | ({study_id} if study_id else set())
    if len(ids) != 1:
        raise ValueError(f"records must be for one patient, got {sorted(ids) or 'none'}")
    sid = ids.pop()
    resolved, conflicts, disagree = resolve(recs)
    ctx = Ctx(resolved, dict(priors or {}), conflicts=conflicts)
    ctx.ai_disagreements = disagree
    dx = _classify(ctx)
    _molecular_extras(ctx, dx)
    _methylation_check(ctx, dx)
    out = _assemble(sid, ctx, dx)
    check_no_priors(out, "integrated diagnosis")
    return out


# ---------------------------------------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------------------------------------

def _dx(status: str, text: str, who: str | None = None, grade: Any = None, suffix: str | None = None,
        favoured: str | None = None) -> dict:
    return {"text": text, "who_type": who, "cns_who_grade": grade, "suffix": suffix, "status": status,
            "favoured": favoured}


def _classify(c: Ctx) -> dict:
    adequacy, tissue, pattern = c.get("adequacy"), c.get("tissue_class"), c.get("glioma_pattern")
    if adequacy.is_("inadequate"):
        c.order("adequacy", "No adequate lesional tissue on the slides", test="additional_sampling")
        return _dx("inadequate", "Insufficient lesional tissue for diagnosis")
    if not c.need("tissue_class", "confirm a glial tumour"):
        return _dx("pending", "Pending: tumour type on H&E", favoured=_favoured_tissue(tissue))
    if not tissue.is_("glioma"):
        c.order("tissue_class", f"Triage call is {_word(tissue.call)}; outside the diffuse glioma rules",
                test="pathologist_review")
        return _dx("not_diffuse_glioma", f"Not a diffuse glioma on triage ({_word(tissue.call)}); "
                                         "for pathologist diagnosis")
    if pattern.is_("circumscribed", "ependymal"):
        c.order("glioma_pattern", f"Growth pattern {pattern.call}; outside the diffuse glioma rules",
                test="pathologist_review")
        return _dx("not_diffuse_glioma", f"Glioma with a {pattern.call} pattern; for pathologist diagnosis")

    h3k27, h3g34, idh = c.get("H3K27"), c.get("H3G34"), c.get("IDH")

    # H3-altered types are defined by the H3 change, whatever the grade-like histology.
    if h3k27.is_("altered"):
        if idh.is_("mutant"):
            c.conflicts.append("H3 K27 alteration and IDH mutation together are unexpected; review both results")
            c.order("H3K27", "Conflict between H3 K27 and IDH", test="pathologist_review")
        c.basis("dmg_def", True, "H3K27")
        c.notes.append("WHO requires a midline location for this type; confirm against the clinical record.")
        # Grade 4 from WHO 2016 as quoted in cIMPACT-NOW update 3.
        c.based_on.append({"criterion": "Diffuse midline glioma, H3 K27-altered corresponds to grade 4",
                           "met": True, "findings": ["H3K27"], "citation": cite("cimpact3")})
        return _dx("complete", TYPES["dmg"], "dmg", 4)
    if h3g34.is_("mutant"):
        c.basis("dhg_def", True, "H3G34")
        c.notes.append("WHO requires a cerebral hemispheric location for this type; confirm against the "
                       "clinical record.")
        return _dx("complete", TYPES["dhg"], "dhg", 4)

    if not c.need("IDH", "separate IDH-mutant from IDH-wildtype gliomas"):
        return _dx("pending", "Pending: diffuse glioma, IDH status awaited", favoured=_favoured_from_idh(c))
    if idh.is_("wildtype") and idh.source == "test" and _is_ihc(idh.rec):
        # R132H immunohistochemistry only detects the commonest IDH1 mutation; negative staining is not
        # proof of IDH-wildtype status, so sequencing decides.
        c.order("IDH", "IDH1 R132H immunohistochemistry is negative; sequence IDH1/IDH2 to exclude other "
                       "mutations", test="IDH_sequencing")
        c.notes.append("IDH-wildtype status rests on negative R132H immunohistochemistry only.")

    if idh.is_("mutant"):
        return _idh_mutant(c)
    return _idh_wildtype(c)


def _idh_mutant(c: Ctx) -> dict:
    codel = c.get("1p19q")
    if not c.need("1p19q", "separate astrocytoma from oligodendroglioma"):
        fav = None
        if c.get("ATRX").is_("lost") or c.get("TP53").is_("mutant"):
            fav = TYPES["astro"]
            c.notes.append("ATRX loss or TP53 mutation favours astrocytoma over oligodendroglioma.")
        elif codel.guess == "codeleted":
            fav = TYPES["oligo"]
        elif codel.guess == "not_codeleted":
            fav = TYPES["astro"]
        return _dx("pending", "Pending: diffuse glioma, IDH-mutant, 1p/19q status awaited", favoured=fav)
    if codel.is_("codeleted"):
        return _oligo(c)
    return _astro(c)


def _histo(c: Ctx, name: str) -> bool | None:
    r = c.get(name)
    return None if r.call is None else r.call == "present"


def _astro(c: Ctx) -> dict:
    c.basis("astro_def", True, "IDH", "1p19q")
    if c.get("ATRX").is_("retained") and c.get("TP53").is_("wildtype"):
        c.notes.append("ATRX retained and TP53 wildtype is less typical of astrocytoma; 1p/19q result rules.")
    nec, mvp = _histo(c, "necrosis"), _histo(c, "microvascular_proliferation")
    cdkn = c.get("CDKN2A")
    hom = None if cdkn.call is None else cdkn.call == "homozygous_deletion"
    g4 = [n for n, v in (("necrosis", nec), ("microvascular_proliferation", mvp), ("CDKN2A", hom)) if v]
    if g4:
        c.basis("astro_g4", True, *g4)
        return _dx("complete", f"{TYPES['astro']}, CNS WHO grade 4", "astro", 4)
    # Grade 4 not shown: every grade 4 criterion must be known to be absent before grading lower.
    missing = []
    for name, v in (("necrosis", nec), ("microvascular_proliferation", mvp)):
        if v is None:
            c.need(name, "exclude CNS WHO grade 4")
            missing.append(name)
    if hom is None:
        c.need("CDKN2A", "exclude CNS WHO grade 4 (p16 immunohistochemistry is not a substitute)")
        missing.append("CDKN2A")
    if missing:
        c.basis("astro_g4", None, *missing)
        return _dx("provisional", f"{TYPES['astro']}, CNS WHO grade pending", "astro", None,
                   favoured=f"{TYPES['astro']}; grade 4 not yet excluded")
    c.basis("astro_g4", False, "necrosis", "microvascular_proliferation", "CDKN2A")
    mit = _histo(c, "mitotic_activity")
    if mit is None:
        c.need("mitotic_activity", "separate grade 2 from grade 3")
        return _dx("provisional", f"{TYPES['astro']}, CNS WHO grade 2 or 3 (pathologist to grade)", "astro",
                   "2 or 3")
    c.basis("astro_g3", mit, "mitotic_activity")
    c.notes.append("Grade 2 versus 3 rests on mitotic activity and anaplasia; confirm the count on the "
                   "whole specimen.")
    grade = 3 if mit else 2
    return _dx("complete", f"{TYPES['astro']}, CNS WHO grade {grade}", "astro", grade)


def _oligo(c: Ctx) -> dict:
    c.basis("oligo_def", True, "IDH", "1p19q")
    if c.get("ATRX").is_("lost"):
        c.conflicts.append("ATRX loss is unusual with 1p/19q codeletion; review both results")
        c.order("ATRX", "ATRX loss with 1p/19q codeletion", test="pathologist_review")
    high = [n for n in ("necrosis", "microvascular_proliferation", "mitotic_activity") if _histo(c, n)]
    c.basis("oligo_grade", None, *high)
    if c.get("CDKN2A").is_("homozygous_deletion"):
        c.notes.append("CDKN2A/B homozygous deletion is reported as a grade 3 marker in oligodendroglioma "
                       "(WHO CNS5 book; not checked against the primary text here).")
        high.append("CDKN2A")
    if high:
        c.notes.append("High-grade features present (" + ", ".join(LABELS[h] for h in high) +
                       "); these favour grade 3.")
        return _dx("provisional", f"{TYPES['oligo']}, CNS WHO grade 2 or 3 (favour 3; pathologist to grade)",
                   "oligo", "2 or 3", favoured=f"{TYPES['oligo']}, CNS WHO grade 3")
    return _dx("provisional", f"{TYPES['oligo']}, CNS WHO grade 2 or 3 (pathologist to grade)", "oligo",
               "2 or 3")


def _idh_wildtype(c: Ctx) -> dict:
    pattern = c.get("glioma_pattern")
    h3_ok = True
    for name in ("H3K27", "H3G34"):
        if not c.need(name, "confirm H3-wildtype, part of the glioblastoma definition"):
            h3_ok = False
    nec, mvp = _histo(c, "necrosis"), _histo(c, "microvascular_proliferation")
    tert, egfr, cn = c.get("TERT"), c.get("EGFR"), c.get("chr7_chr10")
    histo = [n for n, v in (("necrosis", nec), ("microvascular_proliferation", mvp)) if v]
    mol = [n for n, r, hit in (("TERT", tert, "mutant"), ("EGFR", egfr, "amplified"),
                               ("chr7_chr10", cn, "gain7_loss10")) if r.is_(hit)]
    if pattern.is_("oligodendroglial") and mol == ["TERT"] and not histo:
        c.notes.append("TERT promoter mutation is the only glioblastoma criterion and the pattern looks "
                       "oligodendroglial; cIMPACT-NOW 3 asks that a diffuse astrocytic glioma be established "
                       "before using TERT alone.")
        c.order("glioma_pattern", "TERT alone with an oligodendroglial pattern", test="pathologist_review")
    if histo or mol:
        if histo:
            c.basis("gbm_histo", True, *histo)
        if mol:
            c.basis("gbm_mol", True, *mol)
        if not h3_ok:
            c.basis("gbm_def", None, "IDH", "H3K27", "H3G34")
            return _dx("provisional", f"{TYPES['gbm']}, CNS WHO grade 4 (H3 status pending)", "gbm", 4,
                       favoured=TYPES["gbm"])
        c.basis("gbm_def", True, "IDH", "H3K27", "H3G34")
        return _dx("complete", f"{TYPES['gbm']}, CNS WHO grade 4", "gbm", 4)

    # No glioblastoma criterion yet. Order only what is unknown.
    unknown = []
    for name in ("necrosis", "microvascular_proliferation"):
        if c.get(name).call is None:
            c.need(name, "look for the histological criteria of glioblastoma")
            unknown.append(name)
    for name in ("TERT", "EGFR", "chr7_chr10"):
        if c.get(name).call is None:
            c.need(name, "look for the molecular criteria of glioblastoma")
            unknown.append(name)
    if unknown:
        c.basis("gbm_histo", None if {"necrosis", "microvascular_proliferation"} & set(unknown) else False,
                "necrosis", "microvascular_proliferation")
        c.basis("gbm_mol", None if {"TERT", "EGFR", "chr7_chr10"} & set(unknown) else False,
                "TERT", "EGFR", "chr7_chr10")
        fav = TYPES["gbm"] if any(c.get(n).guess in ("present", "mutant", "amplified", "gain7_loss10")
                                  for n in unknown) else None
        return _dx("pending", "Pending: diffuse glioma, IDH-wildtype; glioblastoma criteria not yet complete",
                   favoured=fav)

    c.basis("gbm_histo", False, "necrosis", "microvascular_proliferation")
    c.basis("gbm_mol", False, "TERT", "EGFR", "chr7_chr10")
    if not h3_ok:
        return _dx("pending", "Pending: diffuse glioma, IDH-wildtype, without glioblastoma criteria; H3 status "
                              "awaited")
    meth = c.get("methylation_class")
    if meth.source == "test" and meth.is_("pediatric_hgg_h3wt_idhwt") and h3_ok:
        c.basis("phgg_def", True, "IDH", "H3K27", "H3G34", "methylation_class")
        c.notes.append("Grade 4 for this type is from the WHO CNS5 book; not checked against the primary text "
                       "here.")
        return _dx("complete", TYPES["phgg"], "phgg", 4)
    c.basis("nec", True, "IDH", "TERT", "EGFR", "chr7_chr10")
    c.order("methylation_class", "IDH-wildtype diffuse glioma without glioblastoma criteria; a methylation "
                                 "array can place it (for example paediatric-type or low-grade types)",
            test="methylation_array")
    c.notes.append("Lack of IDH mutation alone does not make a glioma high grade (cIMPACT-NOW update 3).")
    return _dx("provisional", "Diffuse glioma, IDH-wildtype, H3-wildtype, without glioblastoma criteria, NEC",
               None, None, suffix="NEC")


# ---------------------------------------------------------------------------------------------------------
# Extras, cross-checks, assembly
# ---------------------------------------------------------------------------------------------------------

def _molecular_extras(c: Ctx, dx: dict) -> None:
    if dx["who_type"] in ("gbm", "astro") and dx["status"] != "pending" and c.get("MGMT").call is None:
        c.need("MGMT", "report MGMT promoter status for treatment planning",
               blocking=False)


def _methylation_check(c: Ctx, dx: dict) -> None:
    m = c.get("methylation_class")
    if m.rec is None:
        return
    supported = FAMILY_TO_TYPE.get(m.call)
    if m.call in ("control_tissue",):
        c.conflicts.append("Methylation class is control (non-tumour) tissue; check tumour content")
        c.order("methylation_class", "Methylation suggests non-tumour tissue", test="pathologist_review")
        return
    if dx["who_type"] and supported and supported != dx["who_type"]:
        who = "AI" if m.source == "ai" else m.source
        c.conflicts.append(f"Methylation class ({who}) matches {TYPES[supported]}, but the markers give "
                           f"{TYPES[dx['who_type']]}")
        if m.source == "ai":
            c.order("methylation_class", "AI methylation class disagrees with the markers",
                    test="methylation_array")
        c.order("methylation_class", "Methylation class disagrees with the markers", test="pathologist_review")
        if dx["status"] == "complete":
            dx["status"] = "provisional"


def _layer(c: Ctx, names: tuple[str, ...]) -> list[dict]:
    rows = []
    for n in names:
        r = c.f.get(n)
        if r is None or (r.rec is None and r.uncertain is None):
            continue
        rec = r.rec or r.uncertain
        rows.append({
            "finding": n, "call": rec["call"], "source": rec["source"],
            "status": rec["status"] if r.rec else "pending",
            "confidence": rec.get("confidence"),
            "model_id": (rec.get("model") or {}).get("model_id"),
            "method": (rec.get("value") or {}).get("method"),
            "evidence": rec.get("evidence") or [],
            "value": rec.get("value") or {},
        })
    return rows


def _assemble(sid: str, c: Ctx, dx: dict) -> dict:
    if dx["status"] == "complete" and any(o["blocking"] for o in c.orders):
        dx["status"] = "provisional"
    layers = {
        "histology": _layer(c, ("adequacy", "tissue_class", "glioma_pattern")),
        "grade_features": _layer(c, ("necrosis", "microvascular_proliferation", "mitotic_activity")),
        "molecular": _layer(c, ("IDH", "1p19q", "H3K27", "H3G34", "ATRX", "TP53", "CDKN2A", "EGFR",
                                "chr7_chr10", "TERT", "MGMT")),
        "methylation": _layer(c, ("methylation_class",)),
    }
    ai_decisive = sorted({f for b in c.based_on if b["met"] for f in b["findings"]
                          if c.get(f).source == "ai"})
    if ai_decisive:
        c.notes.append("Decided on AI calls without a test or pathologist confirmation: " +
                       ", ".join(LABELS[f] for f in ai_decisive) + ". Confirm on review.")
    out = {
        "contract": DX_CONTRACT,
        "study_id": sid,
        "engine": {"name": "entip-dx", "version": __version__, "rules": RULES_VERSION},
        "integrated_diagnosis": dx,
        "layers": layers,
        "based_on": c.based_on,
        "orders": c.orders,
        "conflicts": c.conflicts,
        "notes": c.notes,
        "ai_decisive": ai_decisive,
        "ai_disagreements": c.ai_disagreements,
        "signoff": {"required": True, "status": "awaiting_pathologist"},
    }
    from .report import render
    out["report_text"] = render(out)
    return out


def _word(call: str | None) -> str:
    if call is None:
        return "not available"
    return CALL_WORDS.get(call, call.replace("_", " "))


def _favoured_tissue(r: Resolved) -> str | None:
    return _word(r.guess) if r.guess else None


def _favoured_from_idh(c: Ctx) -> str | None:
    g = c.get("IDH").guess
    if g == "wildtype":
        return TYPES["gbm"] if any(_histo(c, n) for n in ("necrosis", "microvascular_proliferation")) else None
    if g == "mutant":
        return "IDH-mutant glioma (astrocytoma or oligodendroglioma)"
    return None
