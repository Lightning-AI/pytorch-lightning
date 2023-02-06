# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import json
import pathlib
from typing import Any, Dict, Generic, List, Type, TypeVar

from fastapi import Response, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, parse_obj_as
from pydantic.main import ModelMetaclass

from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.imports import _is_sqlmodel_available

if _is_sqlmodel_available():
    from sqlalchemy.inspection import inspect as sqlalchemy_inspect
    from sqlmodel import JSON, select, Session, SQLModel, TypeDecorator

logger = Logger(__name__)
engine = None

T = TypeVar("T")


# Taken from https://github.com/tiangolo/sqlmodel/issues/63#issuecomment-1081555082
def _pydantic_column_type(pydantic_type: Any) -> Any:
    """This function enables to support JSON types with SQLModel.

    Example::

        from sqlmodel import SQLModel
        from sqlalchemy import Column

        class TrialConfig(SQLModel, table=False):
            ...
            params: Dict[str, Union[Dict[str, float]] = Field(sa_column=Column(pydantic_column_type[Dict[str, float]))
    """

    class PydanticJSONType(TypeDecorator, Generic[T]):
        impl = JSON()

        def __init__(
            self,
            json_encoder=json,
        ):
            self.json_encoder = json_encoder
            super().__init__()

        def bind_processor(self, dialect):
            impl_processor = self.impl.bind_processor(dialect)
            dumps = self.json_encoder.dumps
            if impl_processor:

                def process(value: T):
                    if value is not None:
                        if isinstance(pydantic_type, ModelMetaclass):
                            # This allows to assign non-InDB models and if they're
                            # compatible, they're directly parsed into the InDB
                            # representation, thus hiding the implementation in the
                            # background. However, the InDB model will still be returned
                            value_to_dump = pydantic_type.from_orm(value)
                        else:
                            value_to_dump = value
                        value = jsonable_encoder(value_to_dump)
                    return impl_processor(value)

            else:

                def process(value):
                    if isinstance(pydantic_type, ModelMetaclass):
                        # This allows to assign non-InDB models and if they're
                        # compatible, they're directly parsed into the InDB
                        # representation, thus hiding the implementation in the
                        # background. However, the InDB model will still be returned
                        value_to_dump = pydantic_type.from_orm(value)
                    else:
                        value_to_dump = value
                    value = dumps(jsonable_encoder(value_to_dump))
                    return value

            return process

        def result_processor(self, dialect, coltype) -> T:
            impl_processor = self.impl.result_processor(dialect, coltype)
            if impl_processor:

                def process(value):
                    value = impl_processor(value)
                    if value is None:
                        return None

                    data = value
                    # Explicitly use the generic directly, not type(T)
                    full_obj = parse_obj_as(pydantic_type, data)
                    return full_obj

            else:

                def process(value):
                    if value is None:
                        return None

                    # Explicitly use the generic directly, not type(T)
                    full_obj = parse_obj_as(pydantic_type, value)
                    return full_obj

            return process

        def compare_values(self, x, y):
            return x == y

    return PydanticJSONType


@functools.lru_cache(maxsize=128)  # compatibility for py3.7
def _get_primary_key(model_type: Type["SQLModel"]) -> str:
    primary_keys = sqlalchemy_inspect(model_type).primary_key

    if len(primary_keys) != 1:
        raise ValueError(f"The model {model_type.__name__} should have a single primary key field.")

    return primary_keys[0].name


class _GeneralModel(BaseModel):
    cls_name: str
    data: str
    token: str

    def convert_to_model(self, models: Dict[str, BaseModel]):
        return models[self.cls_name].parse_raw(self.data)

    @classmethod
    def from_obj(cls, obj, token):
        return cls(
            **{
                "cls_name": obj.__class__.__name__,
                "data": obj.json(),
                "token": token,
            }
        )

    @classmethod
    def from_cls(cls, obj_cls, token):
        return cls(
            **{
                "cls_name": obj_cls.__name__,
                "data": "",
                "token": token,
            }
        )


class _SelectAll:
    def __init__(self, models, token):
        print(models, token)
        self.models = models
        self.token = token

    def __call__(self, data: Dict, response: Response):
        if self.token and data["token"] != self.token:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"status": "failure", "reason": "Unauthorized request to the database."}

        with Session(engine) as session:
            cls: Type["SQLModel"] = self.models[data["cls_name"]]
            statement = select(cls)
            results = session.exec(statement)
            return results.all()


class _Insert:
    def __init__(self, models, token):
        self.models = models
        self.token = token

    def __call__(self, data: Dict, response: Response):
        if self.token and data["token"] != self.token:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"status": "failure", "reason": "Unauthorized request to the database."}

        with Session(engine) as session:
            ele = self.models[data["cls_name"]].parse_raw(data["data"])
            session.add(ele)
            session.commit()
            session.refresh(ele)
            return ele


class _Update:
    def __init__(self, models, token):
        self.models = models
        self.token = token

    def __call__(self, data: Dict, response: Response):
        if self.token and data["token"] != self.token:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"status": "failure", "reason": "Unauthorized request to the database."}

        with Session(engine) as session:
            update_data = self.models[data["cls_name"]].parse_raw(data["data"])
            primary_key = _get_primary_key(update_data.__class__)
            identifier = getattr(update_data.__class__, primary_key, None)
            statement = select(update_data.__class__).where(identifier == getattr(update_data, primary_key))
            results = session.exec(statement)
            result = results.one()
            for k, v in vars(update_data).items():
                if k in ("id", "_sa_instance_state"):
                    continue
                if getattr(result, k) != v:
                    setattr(result, k, v)
            session.add(result)
            session.commit()
            session.refresh(result)


class _Delete:
    def __init__(self, models, token):
        self.models = models
        self.token = token

    def __call__(self, data: Dict, response: Response):
        if self.token and data["token"] != self.token:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"status": "failure", "reason": "Unauthorized request to the database."}

        with Session(engine) as session:
            update_data = self.models[data["cls_name"]].parse_raw(data["data"])
            primary_key = _get_primary_key(update_data.__class__)
            identifier = getattr(update_data.__class__, primary_key, None)
            statement = select(update_data.__class__).where(identifier == getattr(update_data, primary_key))
            results = session.exec(statement)
            result = results.one()
            session.delete(result)
            session.commit()


def _create_database(db_filename: str, models: List[Type["SQLModel"]], echo: bool = False):
    global engine

    from sqlmodel import create_engine

    engine = create_engine(f"sqlite:///{pathlib.Path(db_filename).resolve()}", echo=echo)

    logger.debug(f"Creating the following tables {models}")
    try:
        SQLModel.metadata.create_all(engine)
    except Exception as e:
        logger.debug(e)
