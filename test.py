import inspect


def fn(x: float):
    return x


def wrapper(*args, **kwargs):
    print(args, kwargs)


wrapper.__annotations__ = fn.__annotations__
setattr(wrapper, "__signature__", inspect.signature(fn))
print(inspect.signature(wrapper))
print(inspect.signature(fn))

wrapper(1)
