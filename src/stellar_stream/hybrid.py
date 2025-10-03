"""Allows a method to be called as either a classmethod or an instancemethod."""


class hybridmethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls=None):
        """
        When accessed as C.foo, obj is None and cls is the class C.
        When accessed as c.foo, obj is the instance c and cls is C.
        We wrap self.func in a closure so that calling it always
        injects the correct first argument.
        """
        first = obj if obj is not None else cls

        def wrapper(*args, **kwargs):
            return self.func(first, *args, **kwargs)

        return wrapper
