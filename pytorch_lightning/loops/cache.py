from typing import Tuple


class Cache:

    def __init__(self):
        self._store = ...

    def add(self, obj: object, **tags):
        pass

    def merge(self, cache: "Cache"):
        pass

    def filter_by(self, tags: Tuple[str]):
        pass



self.cache = Cache()
self.cache.add("abc", result, batch_idx=, opt_idx=..)
self.cache.add("abc", result, batch_idx=)

self.cache.group_by("abc", ("batch_idx", "opt_idx"))
