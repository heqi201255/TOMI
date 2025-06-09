import time
from functools import wraps
from typing import Any, Union, Type, Tuple, Callable, TypeVar, Optional, overload, List
from types import UnionType


__all__ = ['timer', 'make_property', 'better_property']

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - st:.8f} seconds to execute.")
        return result
    return wrapper

T = TypeVar('T')  # Type variable for the property type


def _create_property_instance(prop_name: str,
                              fget: Optional[Callable[[Any], T]],
                              fset: Optional[Callable[[Any, T], None]],
                              fdel: Optional[Callable],
                              value_type: Optional[Union[Type[Any], Tuple[Type[Any], ...]]],
                              value_options: Optional[List[Any] | Tuple[Any, ...]],
                              value_min: Optional[Union[int, float, Callable]],
                              value_max: Optional[Union[int, float, Callable]],
                              inclusive_boundary: bool,
                              doc: Optional[str],
                              update_func: Optional[Callable[[Any], None]]) -> property:
    """
    Create a property instance given a set of parameters.
    :param prop_name: the name assigned to the property, used for error logging.
    :param fget: Function to retrieve the value, used for comparing new value and current value.
    :param fset: Function to store the value.
    :param fdel: Function to delete the value.
    :param value_type: Expected type of the value.
    :param value_options: A list of valid values of the property. Cannot exist along with value_min or value_max.
    :param value_min: Minimum value of the value.
    :param value_max: Maximum value of the value.
    :param inclusive_boundary: Whether value_min and value_max are inclusive boundaries or exclusive boundaries.
    :param doc: Documentation string for the property.
    :param update_func: Function to call when the value is updated.
    :return: property: A property instance.
    """
    def fset_wrapper(self, value):
        if fget is not None and fget(self) == value:
            return
        if value_type is not None and not isinstance(value, value_type):
            raise TypeError(f"{prop_name} must be of type {value_type}")
        if value_options is not None:
            if value not in value_options:
                raise ValueError(f"{prop_name} must be one of {value_options}")
        if value_min is not None:
            min_v = value_min(self) if callable(value_min) else value_min
            if inclusive_boundary:
                if value < min_v:
                    raise ValueError(f"{prop_name} must be greater than or equal to {min_v}")
            else:
                if value <= min_v:
                    raise ValueError(f"{prop_name} must be greater than {min_v}")
        if value_max is not None:
            max_v = value_max(self) if callable(value_max) else value_max
            if inclusive_boundary:
                if value > max_v:
                    raise ValueError(f"{prop_name} must be less than or equal to {max_v}")
            else:
                if value >= max_v:
                    raise ValueError(f"{prop_name} must be less than {max_v}")
        fset(self, value)
        if update_func is not None:
            update_func(self)
    if fget is None and fset is None:
        raise ValueError("One of fget and fset must not be None to create a valid property.")
    if any(x is not None for x in (value_type, value_min, value_max, update_func)):
        if fset is None:
            raise ValueError("You must specify 'fset' or set 'src_attr_name' to the attribute name to create a setter function.")
        return property(fget,  fset_wrapper, fdel, doc)
    return property(fget, fset, fdel, doc)


def _verify_property_builder_params(params: dict, no_arg_decorator: bool = False) -> None:
    """
    Used by 'make_property' and 'better_property' to check the input parameters type.
    :param params: 'make_property' or 'better_property's input parameters.
    :return: None
    """
    if no_arg_decorator:
        assert callable(params["src_attr_name"])
    else:
        assert params["src_attr_name"] is None or isinstance(params["src_attr_name"], str)
    assert params["fget"] is None or callable(params["fget"])
    assert params["fset"] is None or callable(params["fset"])
    assert params["fdel"] is None or callable(params["fdel"])
    if isinstance(params["value_type"], tuple):
        assert all(isinstance(vt, type) for vt in params["value_type"])
    elif params["value_type"] is not None:
        assert isinstance(params["value_type"], (type, UnionType)) or hasattr(params["value_type"], '__origin__')
    assert params["value_options"] is None or isinstance(params["value_options"], (list, tuple))
    if params["value_options"] is not None and params["value_type"] is not None:
        assert all(isinstance(opt, params["value_type"]) for opt in params["value_options"])
    assert params["value_min"] is None or isinstance(params["value_min"], int | float | Callable)
    assert params["value_max"] is None or isinstance(params["value_max"], int | float | Callable)
    if params["value_options"] is not None:
        assert params["value_min"] is None and params["value_max"] is None, "'value_min' and 'value_max' cannot be set if 'value_options' is set."
    assert params["update_func"] is None or callable(params["update_func"])
    if "doc" in params:
        assert params["doc"] is None or isinstance(params["doc"], str)


def make_property(src_attr_name: Optional[str] = None,
                  fget: Optional[Callable[[Any], T]] = None,
                  fset: Optional[Callable[[Any, T], None]] = None,
                  fdel: Optional[Callable] = None,
                  value_type: Optional[Union[Type[Any], Tuple[Type[Any], ...]]] = None,
                  value_options: Optional[List[Any] | Tuple[Any, ...]] = None,
                  value_min: Optional[Union[int, float, Callable]] = None,
                  value_max: Optional[Union[int, float, Callable]] = None,
                  inclusive_boundary: bool = True,
                  doc: Optional[str] = None,
                  update_func: Optional[Callable[[Any], None]] = None) -> property:
    """
    Creates a property instance for a class attribute with optional validation and update logic.

    Function usage:
        bar = make_property(src_attr_name='_bar', value_type=int, value_min=0)

    :param src_attr_name: Name of the underlying attribute to store the value, if set, will create a default getter & setter function for undefined fget and/or fset functions.
    :param fget: Function to retrieve the value.
    :param fset: Function to store the value.
    :param fdel: Function to delete the value.
    :param value_type: Expected type of the value.
    :param value_options: A list of valid values of the property. Cannot exist along with value_min or value_max.
    :param value_min: Minimum value of the value.
    :param value_max: Maximum value of the value.
    :param inclusive_boundary: Whether value_min and value_max are inclusive boundaries or exclusive boundaries.
    :param doc: Documentation string for the property.
    :param update_func: Function to call when the value is updated.
    :return property: A property instance.
    """
    _verify_property_builder_params(locals())
    f_get = fget if callable(fget) else None if src_attr_name is None else lambda self: getattr(self, src_attr_name)
    f_set = fset if callable(fset) else None if src_attr_name is None else lambda self, value: setattr(self, src_attr_name, value)
    f_del = fdel
    prop_name = src_attr_name.lstrip('_') if src_attr_name is not None else "Value"
    return _create_property_instance(prop_name, f_get, f_set, f_del, value_type, value_options, value_min, value_max, inclusive_boundary, doc, update_func)


@overload
def better_property(
        src_attr_name: Optional[str] = None,
        fget: Optional[Callable[[Any], T]] = None,
        fset: Optional[Callable[[Any, T], None]] = None,
        fdel: Optional[Callable] = None,
        value_type: Optional[Union[Type[Any], Tuple[Type[Any], ...]]] = None,
        value_options: Optional[List[Any] | Tuple[Any, ...]] = None,
        value_min: Optional[Union[int, float, Callable]] = None,
        value_max: Optional[Union[int, float, Callable]] = None,
        inclusive_boundary: bool = True,
        update_func: Optional[Callable[[Any], None]] = None,
) -> Callable[[Callable, bool], property]: ...


@overload
def better_property(
        src_attr_name: Callable[[Any], Union[Callable, Tuple[Callable, ...]]],
        fget: Optional[Callable[[Any], T]] = None,
        fset: Optional[Callable[[Any, T], None]] = None,
        fdel: Optional[Callable] = None,
        value_type: Optional[Union[Type[Any], Tuple[Type[Any], ...]]] = None,
        value_options: Optional[List[Any] | Tuple[Any, ...]] = None,
        value_min: Optional[Union[int, float, Callable]] = None,
        value_max: Optional[Union[int, float, Callable]] = None,
        inclusive_boundary: bool = True,
        update_func: Optional[Callable[[Any], None]] = None,
) -> property: ...


def better_property(src_attr_name: Optional[Union[str, Callable[[Any], Union[Callable, Tuple[Callable, ...]]]]] = None,
                    fget: Optional[Callable[[Any], T]] = None,
                    fset: Optional[Callable[[Any, T], None]] = None,
                    fdel: Optional[Callable] = None,
                    value_type: Optional[Union[Type[Any], Tuple[Type[Any], ...]]] = None,
                    value_options: Optional[List[Any] | Tuple[Any, ...]] = None,
                    value_min: Optional[Union[int, float, Callable]] = None,
                    value_max: Optional[Union[int, float, Callable]] = None,
                    inclusive_boundary: bool = True,
                    update_func: Optional[Callable[[Any], None]] = None) -> Union[Callable[[Callable, bool], property], property]:
    """
    Creates a property instance for a class method with optional validation and update logic.
    Decorator usage:
        @better_property(src_attr_name='_time_signature', value_type=TimeSignature)
        def time_signature(self):
            def fget(self):
                return self._time_signature
            def fset(self, time_signature):
                # Custom logic here
                self._time_signature = time_signature
            return fget, fset
    :param src_attr_name: Name of the underlying attribute to store the value, if set, will create a default getter & setter function for undefined fget and/or fset functions.
    :param fget: Function to retrieve the value.
    :param fset: Function to store the value.
    :param fdel: Function to delete the value.
    :param value_type: Expected type of the value.
    :param value_options: A list of valid values of the property. Cannot exist along with value_min or value_max.
    :param value_min: Minimum value of the value.
    :param value_max: Maximum value of the value.
    :param inclusive_boundary: Whether value_min and value_max are inclusive boundaries or exclusive boundaries.
    :param update_func: Function to call when the value is updated.
    :return: A decorator function.
    """
    params = locals()
    def decorator(func: Callable[[Any], Union[Callable, Tuple[Callable, ...]]], no_arg_decorator: bool = False) -> property:
        """
        Takes in a class method of specific structure and turns it into a property.
        :param func: a class method that should and returns fget and/or fset and/or fdel functions.
        :param no_arg_decorator: whether the decorator "better_property" is used with args (@better_property()) or without args (@better_property).
        :return: A property instance.
        """
        _verify_property_builder_params(params, no_arg_decorator)
        f_get = f_set = f_del = None
        try:
            funcs = func(object())
        except AttributeError:
            raise ValueError("The property method is not designed as a better_property function, you must define inner functions with names of [fget, fset, fdel] and return these functions.")
        funcs = funcs if isinstance(funcs, Tuple) else (funcs,)
        for f in funcs:
            if not callable(f):
                raise ValueError("Returned value must be one or multiple functions.")
            match f.__name__:
                case 'fget': f_get = f
                case 'fset': f_set = f
                case 'fdel': f_del = f
                case _: raise ValueError(f'Function {f.__name__} is not a property method, you must define inner functions with names of [fget, fset, fdel].')
        if fget is not None:
            if f_get is not None:
                raise ValueError("Multiple getter functions found, do not set 'fget' parameter while defining the fget function in the property method.")
            f_get = fget
        elif f_get is None and src_attr_name is not None:
            f_get = lambda self: getattr(self, src_attr_name)

        if fset is not None:
            if f_set is not None:
                raise ValueError("Multiple setter functions found, do not set 'fset' parameter while defining the fset function in the property method.")
            f_set = fset
        elif f_set is None and src_attr_name is not None:
            f_set = lambda self, value: setattr(self, src_attr_name, value)

        if fdel is not None:
            if f_del is not None:
                raise ValueError("Multiple deleter functions found, do not set 'fdel' parameter while defining the fdel function in the property method.")
            f_del = fdel
        prop_name = func.__name__
        return _create_property_instance(prop_name, f_get, f_set, f_del, value_type, value_options, value_min, value_max, inclusive_boundary, func.__doc__, update_func)
    # If the decorator is called without arguments, call the decorator function directly.
    if callable(src_attr_name):
        return decorator(src_attr_name, no_arg_decorator=True)
    return decorator
