import pytest


@pytest.fixture(scope="session")
def live_db():
    """A real central database API (pillar 5) on a free local port, with a write token set.

    Needs the ``entip-db`` package and a PostgreSQL + PostGIS server (ENTIP_TEST_ADMIN_DSN, as in
    db/tests). Skipped when either is missing.
    """
    import os
    import socket
    import threading
    import time
    import uuid

    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("entip_db")
    uvicorn = pytest.importorskip("uvicorn")
    from entip_db.api import create_app
    from entip_db.db import connect, migrate

    admin_dsn = os.environ.get("ENTIP_TEST_ADMIN_DSN", "postgresql://entip:entip@localhost:5432/postgres")
    try:
        admin = psycopg.connect(admin_dsn, autocommit=True)
    except psycopg.OperationalError as e:
        pytest.skip(f"no PostgreSQL server for tests ({e})")
    name = f"entip_test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    dsn = psycopg.conninfo.make_conninfo(admin_dsn, dbname=name)
    conn = connect(dsn)
    migrate(conn)
    conn.close()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    token = "test-write-token"
    server = uvicorn.Server(uvicorn.Config(create_app(dsn, write_token=token, read_token=""), host="127.0.0.1",
                                           port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "token": token, "dsn": dsn}
    server.should_exit = True
    thread.join(timeout=5)
    admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
    admin.close()
