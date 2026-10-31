# Licences: central database (pillar 5)

Checked 31 Oct 2026 from installed package metadata (`License-Expression`) and the Ubuntu PostGIS copyright file.

| Component | Licence | Role | NHS rule |
|---|---|---|---|
| PostgreSQL 16 | PostgreSQL Licence (permissive, BSD-like) | database server | pass |
| PostGIS 3.4 | GPL-2.0-or-later | geometry types and indexes in the server | pass: open source, runs as a server extension; our code does not link to it |
| psycopg 3 (+ psycopg-binary) | LGPL-3.0-only | Python driver | pass: open source, used unmodified as a library |
| FastAPI | MIT | HTTP API | pass |
| Starlette | BSD-3-Clause | used by FastAPI | pass |
| Pydantic | MIT | request models | pass |
| Uvicorn | BSD-3-Clause | HTTP server | pass |
| pytest (test only) | MIT | tests | pass |
| httpx (test only) | BSD-3-Clause | API tests | pass |

No model weights or datasets. The demo patients in `seed.py` are made up.

If a permissive-only driver is ever required, `pg8000` (BSD-3-Clause) could replace psycopg; the
SQL does not depend on the driver.
