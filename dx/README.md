# Integrated diagnosis (AI glioma workflow, step 5)

A fixed WHO CNS5 rules engine. It reads the finding records from steps 1 to 4 (triage, high-grade features,
glioma panel, methylation class) plus real test results, works out the integrated diagnosis, orders the real
test for anything uncertain that would change it, and writes a draft layered report for the pathologist to sign
off. It learns nothing: every call traces to a written criterion with its source. Research use only.

- Shared contract for steps 1 to 4: [`contract/`](contract/README.md).
- Rules and citations: [`src/entip_dx/rules.py`](src/entip_dx/rules.py). Engine: `engine.py`. Report text: `report.py`.

## Run it

```bash
pip install -e 'dx[test]'
entip-dx diagnose records.json --text            # finding records in, draft report out
entip-dx diagnose records.json --priors p.json   # priors (age_years) can change a test request, never the text
entip-dx from-db SYN-0001 --post --text          # read the db API, post the draft as an integrated_dx action
entip-dx validate records.jsonl                  # check a step's output against the contract
entip-dx export-contract dx/contract             # refresh the JSON Schemas after editing contract.py
pytest dx
```

## How it decides

1. A real test beats a pathologist's call, which beats a confident AI call. An uncertain AI call is shown as
   "pending (AI suggests ...)" and its test is ordered. A newer AI run on the same scans replaces the older
   one. Across scans, a high-grade feature counts if any scan shows it, and is absent only when every scan is
   confidently absent; other AI calls that differ between scans are treated as uncertain. Confident AI calls
   that a test overrules are listed for retraining.
2. Negative IDH1 R132H immunohistochemistry is not taken as IDH-wildtype on its own: IDH sequencing is ordered,
   and a sequencing result beats IHC whatever order they arrive in.
3. Inadequate tissue or a non-glioma triage call stops the glioma rules and goes to the pathologist.
4. H3 K27-altered gives diffuse midline glioma; H3 G34-mutant gives diffuse hemispheric glioma (location is
   left for the pathologist to confirm against the clinical record; the report never states it).
5. IDH-mutant: 1p/19q codeleted gives oligodendroglioma (grade 2 or 3, graded by the pathologist); otherwise
   astrocytoma, grade 4 with necrosis, microvascular proliferation or CDKN2A/B homozygous deletion
   (cIMPACT-NOW 5), else grade 3 with significant mitotic activity, else grade 2. A lower grade is only given
   once all grade 4 criteria are known to be absent.
6. IDH-wildtype and H3-wildtype: necrosis, microvascular proliferation, TERT promoter mutation, EGFR
   amplification or +7/-10 gives glioblastoma, grade 4 (WHO CNS5; cIMPACT-NOW 3). Only the criteria still
   unknown are ordered. With none present and H3 confirmed wildtype it is NEC and a methylation array is suggested.
7. A methylation class that disagrees with the markers makes the draft provisional and orders a methylation
   array (if the class was AI) and a pathologist review.
8. MGMT is reported and ordered if missing, but never blocks the diagnosis.
9. An AI-only marker call that defines the type or grade (IDH either way, 1p/19q either way, or any positive
   marker a criterion was met on) gets its real test ordered and the draft stays provisional. AI calls of
   absence for the other markers (for example H3-wildtype) are accepted. `diagnose(..., confirm_ai=False)`
   turns this off, to measure what the AI alone would sign out.
10. Priors are used in one place: negative R132H immunohistochemistry is accepted without sequencing for a
    histological glioblastoma when the age prior is 55 or over (WHO 2016 exception; not checked against the
    primary text here). The report never states the age.

The output (`entip.integrated_dx/1`) lists each criterion as met, not met or pending, the tests to order
(blocking or not), discrepancies, which decisive findings were AI-only, and the draft text. The GUI shows the
latest draft on the patient page.

## Sources

- Louis DN et al. The 2021 WHO Classification of Tumors of the CNS: a summary. Neuro-Oncology 2021;23:1231-51.
  https://doi.org/10.1093/neuonc/noab106 (types, Table 3 grades, Table 4 layered report; full text not read here,
  tables only).
- Brat DJ et al. cIMPACT-NOW update 3. Acta Neuropathol 2018;136:805-10. https://doi.org/10.1007/s00401-018-1913-0
  (full text read: EGFR amplification, +7/-10, TERT promoter criteria and their caveats).
- Brat DJ et al. cIMPACT-NOW update 5. Acta Neuropathol 2020;139:603-8. https://doi.org/10.1007/s00401-020-02127-9
  (full text read: IDH-mutant astrocytoma grades 2, 3, 4 and CDKN2A/B).
- Louis DN et al. cIMPACT-NOW update 1 (NOS and NEC). Acta Neuropathol 2018;135:481-4.
  https://doi.org/10.1007/s00401-018-1808-0 (definitions summarised; text not read here).

Not checked against a primary text and marked as such in the output: CDKN2A/B as a grade 3 marker in
oligodendroglioma, grade 4 for the paediatric-type H3-wildtype IDH-wildtype high-grade glioma, and the
age 55 exception to IDH sequencing.

## Not done yet

- Pathologist sign-off and amendments in the GUI (the draft is read-only there for now).
- Other CNS tumour families (circumscribed gliomas, ependymomas and so on) go to pathologist review.
- Thresholds come from each step's own calibration; the engine trusts the `status` it is given.
