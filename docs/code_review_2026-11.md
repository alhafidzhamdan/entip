# Code review of the five pillars (11 Nov 2026)

Scope: `fork/`, `training/`, `spatial/`, `gui/` and `db/` as on `main` at 74cd83f. The glioma workflow
(PRs #10 to #14) is not covered. Every finding below was checked by reading the code and, where marked,
by running it. All data used was synthetic.

Before any change, every existing test passed (db 33, fork 20, training 14, spatial 8, GUI 5). The spatial
tests had not been run when that pillar was merged; they pass.

## Fixed in this change

| # | Pillar | Finding | How it was confirmed | Fix |
|---|---|---|---|---|
| 1 | spatial | Visium pixel size was 15% too small. The loader used 55 / `spot_diameter_fullres`, but 10x define that field as the pixels across "a theoretical 65µm spot" ([Space Ranger spatial outputs](https://www.10xgenomics.com/support/software/space-ranger/2.1/analysis/outputs/spatial-outputs)). The synthetic data repeated the same assumption, so the tests could not see it. | Read the 10x page. | Use 65. New test checks the scale end to end: neighbouring spots come out 100 µm apart. |
| 2 | db | A retry with the same `action_id` was refused with 422, although the contract says to send your own id "to make retries safe". | Posted the same action twice. | A repeat returns the first result with `replayed: true` (HTTP 200) and changes nothing. Reusing an id for a different action is refused. |
| 3 | db | Several bad inputs gave HTTP 500: `since=garbage`, a negative `limit` or `offset`, a NUL byte in text, a GeoJSON point with one coordinate. | Sent each one. | All return 422 with a plain message. |
| 4 | db | A CHI typed into a free-text field (a note, a site) was accepted. Only key names and a few id fields were checked. | Posted one in `detail`. | Any text value holding a 10-digit number with a valid CHI check digit is refused. Numbers inside hashes or longer codes are not matched. |
| 5 | db | Empty `provenance.actor` was accepted. | Posted `""`. | Refused. |
| 6 | db | No access control at all. | Read the code. | Optional shared tokens: `ENTIP_DB_WRITE_TOKEN`, `ENTIP_DB_READ_TOKEN`. `entip-db serve` refuses a non-local `--host` without a write token. Off by default, so localhost use is unchanged. The GUI sends `dbToken` when set. |
| 7 | fork | Slides were only written to a JSON Lines outbox, with field names that did not match the database. | Read the code. | Registrations use the database's field names and are posted to `POST /actions` with `--db`. The outbox stays as the queue: if the database is down the fork still succeeds, and `entip-fork replay` posts what is waiting. Tested against a real database. |
| 8 | fork | A crash after the key store said "forked" but before the registration was written left a slide that pillar 5 would never hear about (the next run saw "forked" and returned early). | Read the code; new test simulates the crash. | The registration is queued first, then the key store is marked. |
| 9 | fork | A slide with no pixel size was forked with `mpp: null`, which the database refuses. | Read the code. | Refused unless `--mpp` is given. |
| 10 | spatial | Assays, transforms and spots were only written as JSON Lines. | Read the code. | `register --db` writes them to the database as one logged action. Tested against a real database, including a second run replacing the first. |
| 11 | spatial | The assay record carried the full local path of the data folder. Folder names often carry laboratory numbers. | Read the code. | The path is not sent to the database; `--assay-uri` records a location on purpose. |
| 12 | training | Each click got slower with slide size: matching marks to cells compared every mark with every cell. | Timed: 1.4 s per pass for 60,000 cells and 400 marks, run twice per click. | Index by id: 0.02 s. Background sampling no longer builds a cells × positives distance table. |
| 13 | training | Saved heads are loaded with joblib (pickle), from whatever path the Model row holds. Once training reads Model rows from the shared database (PR #7), anyone who can write a Model row could make the service run a file of their choosing. | Read the code. | Heads load only from the service's own `heads/` folder, and only if the file is a live head. |
| 14 | all | No continuous integration. | | `.github/workflows/tests.yml` runs every pillar's tests on each pull request, with PostgreSQL + PostGIS. |

Tests after the change: db 58, fork 28, training 18, spatial 12, GUI 5. All pass.

## Found, not changed: for Al to decide

1. **H-optimus-0 field of view for cells.** `training/cache.py` crops 56 µm around each cell and resizes to
   224 px, which is 0.25 µm/px. The model card says the model "expects images of size 224x224 that were
   extracted at 0.5 microns per pixel" ([H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0)). The
   0.25 µm/px choice follows MitoEd. Whether 112 µm (matching the model card) or 56 µm (tighter on the cell)
   gives better heads is an experiment for the first real-data run, not something to guess.
2. **psycopg licence (LGPL-3.0).** I would keep it. It is open source, used unmodified as a library, which is
   what the LGPL is written to allow, so it passes the NHS open-licence rule as stated. Moving to pg8000 (BSD)
   means rewriting the write path and gives no practical gain. This is my reading, not legal advice.
3. **Retrain gate with small locked sets.** The gate will judge a challenger on as few as 5 locked marks per
   class. Balanced accuracy and calibration error on 10 marks have wide error bars, so a promotion or rejection at
   that size means little. My suggestion: raise `min_locked_per_class` (for example to 50) before any
   real use. Not changed because it alters behaviour Al chose.
4. **A mark can be both locked and trained on.** If the same cell has a locked mark and an unlocked mark, it is
   in both sets. Unlikely in normal use, but nothing prevents it.
5. **The action log pages oldest first.** `GET /actions` and `/patients/{id}/actions` return the oldest rows up
   to `limit`. The GUI asks for 5,000; a patient with more would lose the newest entries.
6. **One database connection per request.** Fine at this scale; a pool is the next step under load.
7. **Names on marks are taken on trust** in the database, training API and GUI. Tokens (finding 6) gate
   services, not people. Real sign-in is NHS single sign-on in a gateway.
8. **Training API has no token of its own**, and `POST /scans/synthetic` is always on.
9. **Spatial training labels** (`training_labels.csv`) are not written to the database as annotations yet.
10. **Three copies of a small HTTP client** (fork, spatial, and training in PR #7). Each pillar is its own
    package, so I left them; a shared `entip-client` package would remove the repeat.
11. **Old pull requests.** #7 (training reads the database), #8 (cross-scan annotations, promote) and #9 (GUI
    live training on database slides) have been open since 31 Oct. This change merges cleanly with each of
    them and with #10 to #14 (trial merges; db and training tests pass with #8 and #7 merged in).
12. **GUI.** Read for escaping of text put into the page: every value goes through `esc()`. No other change
    made, because #9 and #11 are editing the same files.

## Not checked

- Anything on real slides or real assay files. The loaders are written to 10x's documented layouts and tested
  on synthetic files only.
- The H-optimus-0 encoder path (needs the weights and a GPU).
- JPEG-compressed vendor slides in `fork/` (needs `imagecodecs`).
