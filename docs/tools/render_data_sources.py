"""Render docs/data-sources.md from docs/data-sources.csv (the CSV is the source of truth; edit it, then run this).

    python docs/tools/render_data_sources.py
"""
from __future__ import annotations

import csv
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
ORDER = ["in use", "built", "in manifest", "indirect", "planned", "candidate", "not used"]

INTRO = """# Data sources and licences

Every dataset and model weight ENTIP uses or has considered, with its licence or terms. **Edit
`data-sources.csv`, then run `python docs/tools/render_data_sources.py`**; this page is generated from it.

The rule (`docs/pillar_specs.md`): every model, library and dataset must be usable in the NHS. MIT, BSD, Apache and
CC BY are open; non-commercial (NC), no-derivatives (ND) and copyleft items are research options only and must not
reach a deployable model. Software dependencies are listed in each package's `LICENCES.md`.

- **nhs_rule** - `open`: fine for a deployable model. `research only`: comparison or testing, never training a
  deployable model. `check`: terms not confirmed yet, so treat as research only until they are.
- **checked** - when the terms were last looked at. Licences change: re-check before a release.

This is a working register, not legal advice. Edinburgh Innovations and the data protection office have the final
word on commercial use and on personal data (pseudonymised data are still personal data under GDPR).
"""


def _table(rows: list) -> str:
    cols = [("name", "Source"), ("licence_or_terms", "Licence / terms"), ("commercial_use", "Commercial"),
            ("nhs_rule", "NHS rule"), ("status", "Status"), ("used_for", "Used for"), ("access", "Access"),
            ("checked", "Checked"), ("notes", "Notes")]
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: (ORDER.index(r["status"]) if r["status"] in ORDER else 99, r["name"])):
        cells = []
        for k, _ in cols:
            v = (r.get(k) or "").replace("|", "/")
            if k == "name" and r.get("link"):
                v = f"[{v}]({r['link']})"
            if k == "name" and r.get("provider"):
                v += f" ({r['provider']})"
            cells.append(v)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    rows = list(csv.DictReader(open(DOCS / "data-sources.csv", newline="")))
    md = [INTRO, "## Datasets\n", _table([r for r in rows if r["kind"] == "data"]), "",
          "## Model weights and code\n", _table([r for r in rows if r["kind"] == "model"]), ""]
    (DOCS / "data-sources.md").write_text("\n".join(md))
    print(f"wrote {DOCS / 'data-sources.md'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
