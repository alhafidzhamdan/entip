"""Download the trial slides from IDC into ``<out>/<SeriesInstanceUID>/`` (run on a node with internet).

    python stage_slides.py trial.csv slides/

Skips series already downloaded. The GPU job then embeds ``slides/*`` with ``entip-train embed extract``, which
names each bag after its folder, i.e. the SeriesInstanceUID the triage trainer looks up.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main(manifest: str, out: str) -> int:
    from idc_index import IDCClient
    rows = list(csv.DictReader(open(manifest, newline="")))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client, failed = IDCClient(), 0
    for r in rows:
        uid = r["SeriesInstanceUID"]
        if any((out_dir / uid).glob("*.dcm")):
            print(f"have  {uid}")
            continue
        try:
            client.download_from_selection(downloadDir=str(out_dir), seriesInstanceUID=[uid],
                                           dirTemplate="%SeriesInstanceUID", quiet=True, show_progress_bar=False)
            n = len(list((out_dir / uid).glob("*.dcm")))
            if not n:
                raise RuntimeError("no DICOM files after download")
            print(f"got   {uid} ({n} files, {r.get('series_size_MB')} MB)")
        except Exception as e:                      # keep going; the summary says what is missing
            failed += 1
            print(f"FAIL  {uid}: {e!r}"[:300])
    print(f"{len(rows) - failed} of {len(rows)} slides in {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
