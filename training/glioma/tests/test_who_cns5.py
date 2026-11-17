from entip_glioma.triage import who_cns5 as w


def test_tcga_integrated_diagnoses():
    assert w.diffuse_glioma("Mutant", "IDHmut-codel", "G3")[0] == \
        "Oligodendroglioma, IDH-mutant and 1p/19q-codeleted, CNS WHO grade 3"
    assert w.diffuse_glioma("Mutant", "IDHmut-non-codel", "G2")[0] == "Astrocytoma, IDH-mutant, CNS WHO grade 2"
    # IDH-mutant glioblastoma (WHO 2016) is astrocytoma, IDH-mutant, grade 4
    assert w.diffuse_glioma("Mutant", "IDHmut-non-codel", "G4", gbm_study=True)[0] == \
        "Astrocytoma, IDH-mutant, CNS WHO grade 4"
    assert w.diffuse_glioma("WT", "IDHwt", "G4", gbm_study=True)[0] == "Glioblastoma, IDH-wildtype, CNS WHO grade 4"
    # lower-grade histology, IDH-wildtype, TERT promoter mutation: molecular glioblastoma
    t, basis = w.diffuse_glioma("WT", "IDHwt", "G2", tert="Mutant")
    assert t == "Glioblastoma, IDH-wildtype, CNS WHO grade 4" and "TERT" in basis
    assert w.diffuse_glioma("WT", "IDHwt", "G2", tert="WT", chr7_10="No combined CNA")[0] == \
        "Diffuse glioma, IDH-wildtype, NOS (histological grade 2)"
    # CPTAC: IDH only, no 1p/19q
    assert w.diffuse_glioma("Mutant", None, None, gbm_study=True)[0] == "Astrocytoma, IDH-mutant, CNS WHO grade 4"
    assert w.diffuse_glioma(None)[0] == "Diffuse glioma, NOS"


def test_source_terms():
    assert w.from_term("Medulloblastoma, NOS")[0] == "Medulloblastoma, NOS, CNS WHO grade 4"
    assert w.from_term("Pilocytic astrocytoma")[0] == "Pilocytic astrocytoma, CNS WHO grade 1"
    assert w.from_term("Rhabdoid meningioma")[0] == "Meningioma, NOS"          # not AT/RT
    assert w.from_term("Atypical teratoid/rhabdoid tumor")[0].startswith("Atypical teratoid/rhabdoid tumour")
    assert w.from_term("Choroid plexus carcinoma")[0] == "Choroid plexus carcinoma, CNS WHO grade 3"
    assert w.from_term("Glioblastoma")[0] == "High-grade glioma, NOS"           # no IDH: NOS, not a type
    assert w.from_term("Meningeal melanocytoma")[0] == "Meningeal melanocytoma"
    assert w.from_term("Hemangiopericytoma, NOS")[0] == "Solitary fibrous tumour"
    assert w.from_term("") is None and w.from_term("Something unheard of") is None


def test_rows_corrections_first_then_collection_then_molecular():
    corr = {"P1": {"diagnosis": "Meningeal melanoma", "source": "Al Hamdan"}}
    assert w.for_row({"PatientID": "P1", "collection_id": "ccdi_mci", "dx": "Malignant melanoma, NOS"},
                     corrections=corr)[0] == "Meningeal melanoma"
    assert w.for_row({"PatientID": "G", "collection_id": "gtex"})[0] == "Non-neoplastic cerebral cortex"
    assert "stands in for metastasis" in w.for_row({"PatientID": "L", "collection_id": "tcga_luad"})[1]
    mol = {"T": {"IDH_STATUS": "Mutant", "IDH_CODEL_SUBTYPE": "IDHmut-codel", "GRADE": "G2", "SOURCE": "x"}}
    assert w.for_row({"PatientID": "T", "collection_id": "tcga_lgg"}, mol)[0].startswith("Oligodendroglioma, IDH")
    assert w.for_row({"PatientID": "U", "collection_id": "tcga_gbm"}, mol)[0] == "High-grade glioma, NOS"
    t, basis = w.for_row({"PatientID": "X", "collection_id": "ccdi_mci", "dx": "Odd thing", "label": "glioma"})
    assert t == "Gliomas and glioneuronal tumours, NOS" and basis.endswith(": review")
