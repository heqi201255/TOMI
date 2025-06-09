class TOMIEnumDescriptor:
    def __init__(self, _value):
        self._value = _value

    def __get__(self, _, owner):
        return owner(self._value)

    def value(self):
        return self._value


class TOMIEnum:

    def __init__(self, value=None):
        self.options = self.get_options()
        self.name = self.value = 'Unknown'
        if value is not None:
            self.set_value(value)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}.{self.name}'

    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.value == other.value and self.name == other.name

    def __hash__(self):
        return hash((self.__class__.__name__, self.name, self.value))

    @classmethod
    def option_list(cls):
        return list(cls.get_options().keys())

    @classmethod
    def get_options(cls):
        return {name: attr.value() for name, attr in cls.__dict__.items() if isinstance(attr, TOMIEnumDescriptor)}

    @classmethod
    def get_object_by_name(cls, name: str):
        for n, attr in cls.__dict__.items():
            if n == name and isinstance(attr, TOMIEnumDescriptor):
                return cls(attr.value())
        return None

    def set_value(self, value, alias_map: dict = None) -> None:
        if alias_map:
            try:
                value = alias_map[value]
            except KeyError:
                pass
        f = True
        for k, v in self.options.items():
            if value == v:
                f = False
                self.value = v
                self.name = k
        if f:
            raise ValueError(f"{self.__class__.__name__} '{value}' is invalid!")
