
def data_loader(fn):
    """
    Decorator to make any fx with this use the lazy property
    :param fn:
    :return:
    """

    try:
        attr_name = '_lazy_' + fn.__name__
    except Exception as e:
        print(e)

    @property
    def _data_loader(self):
        # lazy init
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))

        # real attr
        return getattr(self, attr_name)
    return _data_loader
