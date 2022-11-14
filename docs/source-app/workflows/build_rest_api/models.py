from pydantic import BaseModel


# 1. Subclass the BaseModel and defines your payload format.
class NamePostConfig(BaseModel):
    name: str
