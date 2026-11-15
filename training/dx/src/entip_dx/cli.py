"""entip-dx command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import contract


def export_contract(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "glioma_step.schema.json").write_text(json.dumps(contract.step_json_schema(), indent=2) + "\n")
    (out / "integrated_dx.schema.json").write_text(json.dumps(contract.dx_json_schema(), indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="entip-dx", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-contract", help="write the JSON Schemas")
    e.add_argument("out", type=Path)
    v = sub.add_parser("validate", help="check a JSON or JSONL file of finding records")
    v.add_argument("path", type=Path)
    a = sub.add_parser("diagnose", help="run the engine on a JSON file of finding records")
    a.add_argument("path", type=Path)
    a.add_argument("--priors", type=Path, help="JSON file with age_years / site / radiology (never reported)")
    a.add_argument("--text", action="store_true", help="print only the draft report")
    s = sub.add_parser("from-db", help="run the engine on a patient in the ENTIP database API")
    s.add_argument("study_id")
    s.add_argument("--api", default="http://127.0.0.1:8705")
    s.add_argument("--post", action="store_true", help="post the draft back as an integrated_dx action")
    s.add_argument("--text", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "export-contract":
        export_contract(args.out)
        return 0
    if args.cmd == "validate":
        recs = _load(args.path)
        bad = 0
        for i, r in enumerate(recs):
            try:
                contract.validate_finding(r)
            except contract.ContractError as exc:
                bad += 1
                print(f"record {i}: {exc}", file=sys.stderr)
        print(f"{len(recs) - bad} valid, {bad} invalid")
        return 1 if bad else 0
    from . import engine
    if args.cmd == "diagnose":
        priors = json.loads(args.priors.read_text()) if args.priors else None
        out = engine.diagnose(_load(args.path), priors=priors)
    else:
        from . import dbio
        recs = dbio.fetch_records(args.api, args.study_id)
        out = engine.diagnose(recs, study_id=args.study_id)
        if args.post:
            dbio.post_dx(args.api, out)
    print(out["report_text"] if args.text else json.dumps(out, indent=2))
    return 0


def _load(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data.get("records", [data])


if __name__ == "__main__":
    raise SystemExit(main())
