from lightning_app.utilities.imports import _is_sql_model_available

if _is_sql_model_available():
    from sqlalchemy.future import Engine as _FutureEngine
    from sqlmodel import create_engine, SQLModel


def _create_sqlite_engine(db_name: str = "database.db", debug: bool = False) -> "_FutureEngine":
    engine = create_engine(f"sqlite:///{db_name}", echo=debug)
    SQLModel.metadata.create_all(engine)
    return engine
