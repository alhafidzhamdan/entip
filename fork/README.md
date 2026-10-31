# fork: slide fork and link (pillar 1)

Research use only. Synthetic or public data only; no patient data in this repository.

Each whole-slide image is split into two copies that share one random **linkage code** (e.g. `L-9J7KMKRTW`):

| | NHS side (`--nhs-root`) | Research side (`--research-root`) |
|---|---|---|
| Slide | original file, byte for byte, plus `entip_link.json` with the linkage code | `slides/<link_code>.tiff`: pyramid only, rebuilt metadata |
| Identifiers | CHI, accession, original file name | none: study_id, specimen/block/section IDs, linkage code |
| Records | `keystore.sqlite` (ID map + audit log, file mode 600) | `outbox/p5.jsonl` (objects for the central database), `actions.jsonl` |

The two roots must be separate folders; the tool refuses if one sits inside the other. CHI never reaches the research side.

## What happens on `fork`

1. Hash the source file (SHA-256). The same file forked twice gets the same linkage code.
2. Mint or look up IDs in the key store: one `study_id` per CHI, one specimen ID per accession, then block, section and scan (linkage code). IDs are random, so they cannot be reversed without the key store, and end with a check character so typos are caught.
3. Copy the clinical file unchanged and write the linkage code beside it.
4. Write the research copy: image pyramid only, tile by tile. Label, macro and thumbnail images are dropped; the description is rebuilt from an allow-list (µm/px, magnification, levels, IDs, stain). Vendor fields (file name, barcode, user, date, scanner ID), private tags, XMP and dates are not carried over. Pixels are kept exactly (lossless zlib by default).
5. Scan the research copy for the case's identifiers (CHI, accession, patient name, file name, in several spellings and encodings) and for any CHI-shaped number in its tags. On any finding the copy is deleted and the fork is refused.
6. Hand Patient, Specimen, Block, Section and Scan objects with provenance to pillar 5, and log the action.

Re-identification (`reidentify`) runs on the NHS side only, needs a reason, and is written to the audit log.

## Quick start

```bash
pip install -e ".[test]"
entip-fork synth demo/in/case1.svs                     # prints its synthetic CHI and accession
entip-fork fork demo/in/case1.svs --nhs-root demo/nhs --research-root demo/research \
    --chi <chi> --accession <accession> --block A1 --section L1 --stain "H&E"
entip-fork batch manifest.csv --nhs-root demo/nhs --research-root demo/research
entip-fork reidentify L-XXXXXXXXX --nhs-root demo/nhs --reason "MDT query"
entip-fork check demo/research/slides/L-XXXXXXXXX.tiff --identifier <chi>
entip-fork licences
pytest
```

Manifest columns: `slide,chi,accession,block,section,stain[,patient_name,extra_identifiers]` (extra identifiers separated by `;`).

## Safety rails

- CHI must pass the format and check-digit test (Modulus-11, or Modulus-10/Luhn used for some new CHIs from August 2026, per [Public Health Scotland's phsmethods notes](https://code.publichealthscotland.scot/phsmethods/articles/chi-operations.html)).
- Any CHI whose first six digits are a real date of birth is refused unless `--allow-real-identifiers` is passed. That flag is for use only after governance approval. Synthetic CHIs use day `00`, which is not a real date.
- An accession cannot be attached to two patients, and one file cannot be filed under two sections.

## Limits (next steps)

- Formats: anything tifffile reads as a pyramid (Aperio SVS, generic pyramidal TIFF, NDPI, OME-TIFF). JPEG-compressed vendor slides need `pip install ".[jpeg]"` (imagecodecs, BSD-3). MRXS and iSyntax are not handled yet.
- Text burned into the scanned tissue area (pen marks, stickers) is not detected.
- Z-stacks and multichannel fluorescence are refused for now.
- Object field names are placeholders until the pillar 5 schema lands; then `registry.py` maps to it and posts to the P5 API instead of the outbox.

## Licences

This package follows the repository licence (PolyForm Noncommercial 1.0.0, see [../LICENSE.md](../LICENSE.md)). Its dependencies:

| Package | Licence | Use |
|---|---|---|
| numpy | BSD-3-Clause | required |
| tifffile | BSD-3-Clause | required |
| sqlite3 | Python standard library (PSF-2.0); SQLite is public domain | required |
| imagecodecs | BSD-3-Clause | optional, JPEG vendor slides |
| pytest | MIT | tests only |
