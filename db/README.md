# Central database (pillar 5)

PostgreSQL + PostGIS schema for the ENTIP ontology, an append-only action log, a one-row-per-patient view and
an HTTP API. This is the contract the other pillars code against: see [`schema/README.md`](schema/README.md).

Research use only. Synthetic or public data only; the demo patients are made up.

## Run it

```bash
docker compose up -d                 # PostgreSQL 16 + PostGIS 3.4 on localhost:5432 (or use your own server)
pip install -e '.[test]'
export ENTIP_DB_DSN=postgresql://entip:entip@localhost:5432/entip_dev
entip-db migrate                     # apply db/src/entip_db/migrations/*.sql
entip-db seed                        # three synthetic patients
entip-db serve                       # API on http://127.0.0.1:8705 (docs at /docs)
entip-db patient SYN-0001            # print a patient row and tree
entip-db export-schema schema        # refresh schema/ after a migration
pytest                               # needs a server; set ENTIP_TEST_ADMIN_DSN if not the default
```

## Layout

- `src/entip_db/migrations/`: numbered SQL migrations (the schema itself).
- `db.py`: connections, migrations, payload checks and `apply_action`, the only write path.
- `queries.py`: patient row and tree, slide objects, the log.
- `api.py`: FastAPI app. `contract.py`: the machine-readable contract. `seed.py`: synthetic patients.

## Not done yet

- Authentication and roles (reader, writer, admin). Run on localhost or behind the platform gateway for now.
- Bulk loading through `COPY` for millions of cells (the JSON path handles a few hundred thousand per call).
- Embedding cache (pillar 2 keeps it on disk; the database could index it later).
