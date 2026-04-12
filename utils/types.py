"""Type inspection utilities.

Exposes:
- is_method: Check if object has a method (not property)
- is_property: Check if object has a property
"""

import inspect


def is_method(obj, name: str) -> bool:
    """Check if object has a method with given name.

    Args:
        obj: Object to check.
        name: Method name to look for.

    Returns:
        True if obj has a function or method descriptor for name.
    """
    attr = getattr(type(obj), name, None)
    return inspect.isfunction(attr) or inspect.ismethoddescriptor(attr)


def is_property(obj, name: str) -> bool:
    """Check if object has a property with given name.

    Args:
        obj: Object to check.
        name: Property name to look for.

    Returns:
        True if obj has a property for name.
    """
    return isinstance(getattr(type(obj), name, None), property)
