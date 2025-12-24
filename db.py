import os
from sqlmodel import SQLModel, create_engine, Session

DEFAULT_DB_URL = "postgresql+psycopg://hitl:hitl_password@127.0.0.1:5432/hitl_db"

raw_url = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Normalize common issues:
# - extra whitespace/newlines
# - old "postgres://" scheme
# - ensure we use psycopg (v3) driver: postgresql+psycopg://
DATABASE_URL = raw_url.strip()

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

engine = create_engine(DATABASE_URL, echo=False)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
