"""A collection of functional programming tools.

This module contains a collection of tools for functional programming.
"""
from __future__ import annotations

from functools import partial, reduce
from itertools import count
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
)

T = TypeVar("T")
V = TypeVar("V")
K = TypeVar("K", bound=Hashable)
VK = TypeVar("VK", bound=Hashable)


def prefix_keys(d: Mapping[str, V], prefix: str) -> dict[str, V]:
    """Prefix the keys of a mapping.

    ```python exec="true" source="material-block" result="python" title="prefix_keys"
    from byop.functional import prefix_keys

    d = {"a": 1, "b": 2}
    print(prefix_keys(d, "c:"))
    ```
    """
    return {prefix + k: v for k, v in d.items()}


def mapping_select(d: Mapping[str, V], prefix: str) -> dict[str, V]:
    """Select a subset of a mapping.

    ```python exec="true" source="material-block" result="python" title="mapping_select"
    from byop.functional import mapping_select

    d = {"a:b:c": 1, "a:b:d": 2, "c:elephant": 3}
    print(mapping_select(d, "a:b:"))
    ```

    Args:
        d: The mapping to select from.
        prefix: The prefix to select.

    Returns:
        The selected subset of the mapping.
    """
    return {k[len(prefix) :]: v for k, v in d.items() if k.startswith(prefix)}


def reverse_enumerate(
    seq: Sequence[T],
    start: int | None = None,
) -> Iterator[tuple[int, T]]:
    """Reverse enumerate.

    This function is similar to enumerate, but it iterates over the
    sequence in reverse.

    ```python exec="true" source="material-block" result="python" title="reverse_enumerate"
    from byop.functional import reverse_enumerate

    xs = ["a", "b", "c"]
    for i, x in reverse_enumerate(xs):
        print(i, x)
    ```

    Args:
        seq: The sequence to iterate over.
        start: The starting index.

    Returns:
        An iterator over the sequence.
    """  # noqa: E501
    if start is None:
        start = len(seq) - 1
    yield from zip(count(start, -1), reversed(seq))


def rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    """Recursive version of getattr.

    This function is similar to getattr, but it allows you to get
    attributes using '.' notation.

    ```python exec="true" source="material-block" result="python" title="rgetattr"
    from byop.functional import rgetattr

    class A:
        x = 1

    class B:
        a = A()

    b = B()
    print(rgetattr(b, "a.x"))
    ```

    See Also:
        * https://stackoverflow.com/a/31174427/5332072

    Args:
        obj: The object to get the attribute from.
        attr: The attribute to get.
        *args: The default value to return if the attribute is not found.

    Returns:
        The attribute.
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj, *attr.split(".")])


def funcname(func: Callable, default: str | None = None) -> str:
    """Get the name of a function.

    Args:
        func: The function to get the name of.
        default: The default value to return if the name cannot be
            determined automatically.

    Returns:
        The name of the function.
    """
    if isinstance(func, partial):
        return func.func.__name__
    if hasattr(func, "__qualname__"):
        return func.__qualname__
    if hasattr(func, "__class__"):
        return func.__class__.__name__
    if default is not None:
        return default
    return str(func)


def callstring(f: Callable, *args: Any, **kwargs: Any) -> str:
    """Get a string representation of a function call.

    Args:
        f: The function to get the string representation of.
        *args: The positional arguments.
        **kwargs: The keyword arguments.

    Returns:
        The string representation of the function call.
    """
    # Iterate over all args, convert them to str, and join them
    args_str = ""
    if any(args):
        args_str += ", ".join(map(str, args))
    if any(kwargs):
        args_str += ", ".join(f"{k}={v}" for k, v in kwargs.items())

    return f"{funcname(f)}({args_str})"


class Flag(Generic[T]):
    """A flag.

    This class is used to store a value that can be reset to its
    initial value.

    ```python
    flag = Flag(1)
    flag.value  # 1

    flag.set(2)
    flag.value  # 2

    flag.reset()
    flag.value  # 1
    ```

    Args:
        initial: The initial value.

    Attributes:
        value: The current flag value.
        initial: The initial value.
    """

    def __init__(self, initial: T) -> None:
        """Initialize the flag."""
        self.value = initial
        self.initial = initial

    def reset(self) -> None:
        """Reset the flag to its initial value."""
        self.value = self.initial

    def set(self, value: T) -> None:
        """Set the flag value."""
        self.value = value

    def __bool__(self) -> bool:
        """Get the flag value."""
        return bool(self.value)
