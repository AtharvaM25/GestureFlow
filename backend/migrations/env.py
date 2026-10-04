from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from backend import models  # noqa: F401  (registers the tables on Base.metadata)
from backend.config import get_settings
from backend.db import Base

if context.config.config_file_name is not None:
    # keep the app's own loggers (e.g. gestureflow.access) working when migrations run in-process
    fileConfig(context.config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=get_settings().database_url, target_metadata=target_metadata,
                      literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    with create_engine(get_settings().database_url).connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
