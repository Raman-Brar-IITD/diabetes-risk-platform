"""SQLAlchemy engine/session setup.

Reads DATABASE_URL from st.secrets when a Streamlit runtime is active,
falling back to the environment variable otherwise, so this module (and
anything built on repository.py) works the same in tests and scripts
with no Streamlit involved.
"""
import os
import threading

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.persistence.models import Base

_init_lock = threading.Lock()


def _database_url():
    try:
        import streamlit as st
        if "DATABASE_URL" in st.secrets:
            return st.secrets["DATABASE_URL"]
    except Exception:
        pass
    return os.environ.get("DATABASE_URL")


_engine = None
_SessionLocal = None


def is_configured() -> bool:
    """Whether a DATABASE_URL is available. Doesn't open a connection."""
    return bool(_database_url())


def get_engine():
    global _engine
    if _engine is None:
        url = _database_url()
        if not url:
            return None
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


# Columns added after the first release. create_all() never alters a table
# that already exists (e.g. the live Supabase one), so these are added in
# place when missing. Names/types are constants — nothing user-supplied is
# ever interpolated into the DDL below.
ADDED_COLUMNS = {
    "assessments": {
        "explanation_text": "TEXT",
        "recommendation_text": "TEXT",
        "top_factors_json": "JSON",
        "reviewed": "BOOLEAN DEFAULT FALSE",
        "reviewed_by": "VARCHAR",
        "reviewed_at": "TIMESTAMP",
        "review_note": "TEXT",
    },
}


def ensure_columns(engine):
    """Adds any ADDED_COLUMNS missing from existing tables. Idempotent."""
    inspector = inspect(engine)
    if_not_exists = "IF NOT EXISTS " if engine.dialect.name == "postgresql" else ""
    for table, columns in ADDED_COLUMNS.items():
        if not inspector.has_table(table):
            continue
        existing = {c["name"] for c in inspector.get_columns(table)}
        for name, ddl_type in columns.items():
            if name in existing:
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {if_not_exists}{name} {ddl_type}"))
            except SQLAlchemyError:
                # Another process added it between the check and the ALTER.
                pass


def init_db(engine):
    """Creates missing tables and columns. Additive and idempotent — safe to call every startup."""
    try:
        Base.metadata.create_all(engine)
    except SQLAlchemyError:
        # Concurrent first-time creation race — harmless if the tables exist.
        pass
    ensure_columns(engine)


def get_session():
    """Returns a new Session, or None if DATABASE_URL isn't configured."""
    global _SessionLocal
    engine = get_engine()
    if engine is None:
        return None
    if _SessionLocal is None:
        with _init_lock:
            if _SessionLocal is None:
                try:
                    init_db(engine)
                except SQLAlchemyError:
                    # Two concurrent Streamlit sessions can both see "table
                    # doesn't exist yet" and race to create it — harmless as
                    # long as the tables end up existing either way.
                    pass
                _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()
