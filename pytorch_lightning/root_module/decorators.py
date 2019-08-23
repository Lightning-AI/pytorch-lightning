
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
        try:
            if not hasattr(self, attr_name):
                setattr(self, attr_name, fn(self))
        except AttributeError as e:
            print(e)
            raise e

        # return already init value
        try:
            attr = getattr(self, attr_name)
            return attr
        except AttributeError as e:
            raise e

    return _data_loader
