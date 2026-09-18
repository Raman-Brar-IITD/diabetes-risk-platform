"""SQLAlchemy engine/session setup.

Reads DATABASE_URL from st.secrets when a Streamlit runtime is active,
falling back to the environment variable otherwise, so this module (and
anything built on repository.py) works the same in tests and scripts
with no Streamlit involved.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.persistence.models import Base


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


def get_engine():
    global _engine
    if _engine is None:
        url = _database_url()
        if not url:
            return None
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def init_db(engine):
    """Creates any missing tables. Additive and idempotent — safe to call every startup."""
    Base.metadata.create_all(engine)


def get_session():
    """Returns a new Session, or None if DATABASE_URL isn't configured."""
    global _SessionLocal
    engine = get_engine()
    if engine is None:
        return None
    if _SessionLocal is None:
        init_db(engine)
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()
