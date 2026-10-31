"""entip-db: migrate, seed, serve, export the schema contract."""
from __future__ import annotations

import argparse
import json
import sys

from .db import connect, dsn, migrate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="entip-db", description="ENTIP central database (pillar 5)")
    ap.add_argument("--dsn", default=None, help="PostgreSQL URL (default: $ENTIP_DB_DSN or local entip_dev)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="apply schema migrations")
    s = sub.add_parser("seed", help="add the synthetic demo patients")
    s.add_argument("--cells", type=int, default=400)
    e = sub.add_parser("export-schema", help="write schema.json, SQL and openapi.json")
    e.add_argument("out_dir")
    v = sub.add_parser("serve", help="run the HTTP API")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8705)
    p = sub.add_parser("patient", help="print one patient row and tree as JSON")
    p.add_argument("study_id")
    a = ap.parse_args(argv)
    target = a.dsn or dsn()

    if a.cmd == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(target), host=a.host, port=a.port)
        return 0

    conn = connect(target)
    try:
        if a.cmd == "migrate":
            done = migrate(conn)
            print("applied: " + (", ".join(done) if done else "nothing (up to date)"))
        elif a.cmd == "seed":
            from .seed import seed
            r = seed(conn, n_cells=a.cells)
            print(f"seeded {len(r['patients'])} synthetic patients in {len(r['actions'])} actions")
        elif a.cmd == "export-schema":
            from .api import create_app
            from .contract import export
            for f in export(conn, a.out_dir, app=create_app(target)):
                print(f)
        elif a.cmd == "patient":
            from .queries import patient_tree
            t = patient_tree(conn, a.study_id)
            if t is None:
                print("not found", file=sys.stderr)
                return 1
            print(json.dumps(t, indent=2, default=str))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
