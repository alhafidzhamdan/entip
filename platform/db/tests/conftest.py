"""Tests run against a real PostgreSQL + PostGIS server.

Set ENTIP_TEST_ADMIN_DSN to a URL of a role that can create databases
(default: postgresql://entip:entip@localhost:5432/postgres). Each test session
creates a fresh database and drops it afterwards. Skipped when no server.
"""
import os
import uuid

import psycopg
import pytest

ADMIN = os.environ.get("ENTIP_TEST_ADMIN_DSN", "postgresql://entip:entip@localhost:5432/postgres")


@pytest.fixture(scope="session")
def dsn():
    try:
        admin = psycopg.connect(ADMIN, autocommit=True)
    except psycopg.OperationalError as e:
        pytest.skip(f"no PostgreSQL server for tests ({e})")
    name = f"entip_test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    url = psycopg.conninfo.make_conninfo(ADMIN, dbname=name)
    yield url
    admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture(scope="session")
def seeded(dsn):
    from entip_db.db import connect, migrate
    from entip_db.seed import seed
    conn = connect(dsn)
    migrate(conn)
    seed(conn, n_cells=200)
    conn.close()
    return dsn


@pytest.fixture()
def conn(seeded):
    from entip_db.db import connect
    c = connect(seeded)
    yield c
    c.close()


@pytest.fixture(scope="session")
def client(seeded):
    from fastapi.testclient import TestClient
    from entip_db.api import create_app
    return TestClient(create_app(seeded))
