"""Write gui/web/mock/patients.json: five invented patients in the ENTIP ontology.

Run after make_mock_slides.py (it reads slides/summary.json for the mock MitoEd outputs).
Every name, id, date and result here is made up. Study ids are pseudonymised research ids;
the clinical link lives in the NHS-side key store (P1) and is never in this file.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "web" / "mock"

PEOPLE = [
    dict(study_id="ENT-0001", scan="SCN-0001", sex="F", age_band="60-69", site="Right temporal lobe",
         diagnosis="Glioblastoma, IDH-wildtype", who_grade=4, status="In review",
         molecular={"IDH": "wildtype", "1p/19q": "not tested", "MGMT": "methylated", "CDKN2A/B": "homozygous deletion",
                    "TERT promoter": "C228T", "EGFR": "amplified", "+7/-10": "present"},
         methylation=("Glioblastoma, IDH-wildtype, RTK II", 0.96)),
    dict(study_id="ENT-0002", scan="SCN-0002", sex="M", age_band="30-39", site="Left frontal lobe",
         diagnosis="Astrocytoma, IDH-mutant", who_grade=3, status="New",
         molecular={"IDH": "IDH1 R132H", "1p/19q": "intact", "ATRX": "loss", "MGMT": "methylated",
                    "CDKN2A/B": "retained", "TP53": "mutant"},
         methylation=("Astrocytoma, IDH-mutant, high grade", 0.91)),
    dict(study_id="ENT-0003", scan="SCN-0003", sex="F", age_band="40-49", site="Right frontal lobe",
         diagnosis="Oligodendroglioma, IDH-mutant and 1p/19q-codeleted", who_grade=2, status="Reported",
         molecular={"IDH": "IDH1 R132H", "1p/19q": "codeleted", "ATRX": "retained", "TERT promoter": "C250T",
                    "MGMT": "methylated"},
         methylation=("Oligodendroglioma, IDH-mutant and 1p/19q-codeleted", 0.97)),
    dict(study_id="ENT-0004", scan="SCN-0004", sex="M", age_band="20-29", site="Left insula",
         diagnosis="Astrocytoma, IDH-mutant", who_grade=2, status="New",
         molecular={"IDH": "IDH1 R132H", "1p/19q": "intact", "ATRX": "loss", "CDKN2A/B": "retained"},
         methylation=None),
    dict(study_id="ENT-0005", scan="SCN-0005", sex="M", age_band="70-79", site="Left parietal lobe",
         diagnosis="Glioblastoma, IDH-wildtype", who_grade=4, status="In review",
         molecular={"IDH": "wildtype", "MGMT": "unmethylated", "TERT promoter": "C228T", "CDKN2A/B": "retained",
                    "EGFR": "not amplified", "+7/-10": "present"},
         methylation=("Glioblastoma, IDH-wildtype, mesenchymal", 0.88)),
]

FEATURES = [
    {"name": "Mitotic figure", "level": "cell", "description": "Cells in mitosis. Seeded from MitoEd.", "source": "mitoed"},
    {"name": "Apoptotic body", "level": "cell", "description": "Shrunken dense nuclei with a clear halo. A common mitosis mimic.", "source": "user"},
    {"name": "Endothelial cell", "level": "cell", "description": "Vessel lining cells, for microvascular proliferation.", "source": "user"},
]


def main():
    summary = json.loads((ROOT / "slides" / "summary.json").read_text())
    patients = []
    for i, p in enumerate(PEOPLE, start=1):
        s = summary.get(p["scan"])
        spec = f"SP-{i:04d}-1"
        blk = f"{spec}-A1"
        outputs = []
        if s:
            est = s.get("n_mitoed_called", s["n_mitosis"])
            outputs.append({
                "output_id": f"MO-{i:04d}-1", "object_type": "scan", "object_id": p["scan"],
                "measure": "Mitotic figures per mm²", "value": round(est / s["tissue_area_mm2"], 1),
                "detail": f"{est} figures called in {s['tissue_area_mm2']} mm² of tissue",
                "source": {"kind": "model", "name": "MitoEd", "version": "0.5.4 (mock output)"},
                "ts": f"2026-10-{20 + i}T09:00:00Z"})
        sections = [{
            "section_id": f"{blk}-L1", "level": 1, "stain": "H&E", "thickness_um": 4,
            "scans": [{"scan_id": p["scan"], "stain": "H&E", "scanner": "synthetic", "objective": "40x",
                       "mpp": 0.25, "width_px": 6144, "height_px": 4608,
                       "copy": "research", "linkage": "held in NHS key store",
                       "tile_source": f"slides/{p['scan']}/slide.dzi" if s else None,
                       "source": {"kind": "model", "name": "make_mock_slides", "version": "1"}}],
            "assays": [],
        }, {
            "section_id": f"{blk}-L2", "level": 2, "stain": "Ki-67", "thickness_um": 4,
            "scans": [{"scan_id": f"{p['scan']}-K", "stain": "Ki-67", "scanner": "synthetic", "objective": "40x",
                       "mpp": 0.25, "copy": "research", "linkage": "held in NHS key store", "tile_source": None,
                       "note": "Listed only; no mock image."}],
            "assays": [],
        }]
        if p["methylation"]:
            sections.append({"section_id": f"{blk}-C1", "level": 3, "stain": "curls", "thickness_um": 10, "scans": [],
                             "assays": [{"assay_id": f"AS-{i:04d}-M", "kind": "DNA methylation array",
                                         "platform": "EPIC v2", "status": "complete",
                                         "result": f"{p['methylation'][0]} (calibrated score {p['methylation'][1]})",
                                         "source": {"kind": "model", "name": "methylation classifier", "version": "mock"}}]})
        if i in (1, 3):
            sections.append({"section_id": f"{blk}-V1", "level": 4, "stain": "H&E (spatial)", "thickness_um": 5, "scans": [],
                             "assays": [{"assay_id": f"AS-{i:04d}-V", "kind": "Spatial transcriptomics",
                                         "platform": "Visium HD", "status": "planned", "result": None,
                                         "transform": None}]})
        patients.append({
            "study_id": p["study_id"], "sex": p["sex"], "age_band": p["age_band"],
            "diagnosis": p["diagnosis"], "who_grade": p["who_grade"], "review_status": p["status"],
            "molecular": p["molecular"], "model_outputs": outputs,
            "specimens": [{"specimen_id": spec, "procedure": "Resection", "site": p["site"],
                           "received": f"2026-10-{10 + i:02d}",
                           "blocks": [{"block_id": blk, "label": "A1", "sections": sections}]}],
        })
    out = {"note": "Synthetic data for the ENTIP GUI mock. Not real patients.", "features": FEATURES, "patients": patients}
    (ROOT / "patients.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {len(patients)} patients")


if __name__ == "__main__":
    main()
