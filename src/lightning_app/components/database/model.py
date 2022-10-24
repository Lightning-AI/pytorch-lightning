from typing import Any, Optional

from lightning_app.utilities.imports import _is_sqlmodel_available

if _is_sqlmodel_available():
    import sqlalchemy
    from sqlmodel import Field, select, Session, SQLModel
else:
    SQLModel = object


def LightningField(*args, **kwargs):
    id = kwargs.pop("id", None)
    assert isinstance(id, int)
    return Field(*args, **kwargs)


class LightningSQLModel(SQLModel):

    __table_args__ = {"extend_existing": True}
    _client: Optional[Any] = None
    component_name: Optional[str]
    _should_reload: bool = False
    _allowed: bool = False

    def __setattr__(self, key, value) -> None:
        if key in ("_client", "component_name", "_should_reload", "_allowed"):
            object.__setattr__(self, key, value)
            return

        if self._client:
            with Session(self._engine) as session:
                statement = select(self.__class__).where(
                    getattr(self.__class__, "component_name") == self.component_name
                )
                results = session.exec(statement)
                items = results.all()
                if not items:
                    super().__setattr__(key, value)
                    session.add(self)
                    session.commit()
                    session.refresh(self)
                elif len(items) > 1:
                    raise Exception("Each entry in the database should be unique.")
                else:
                    item = items[0]
                    setattr(item, key, value)
                    session.add(item)
                    session.commit()
                    session.refresh(item)
                    object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)
            if key in self.__annotations__:
                breakpoint()

    def reload(self):
        self._should_reload = True

    def _reload(self):
        assert self.component_name
        with Session(self._engine) as session:
            statement = select(self.__class__).where(getattr(self.__class__, "component_name") == self.component_name)
            results = session.exec(statement)
            items = results.all()
            if len(items) == 1:
                item = items[0]
                for k, v in vars(item).items():
                    object.__setattr__(self, k, v)
            elif len(items) > 1:
                raise Exception("Each entry in the database should be unique.")
