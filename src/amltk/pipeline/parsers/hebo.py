"""TODO."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from hebo.design_space.design_space import DesignSpace

if TYPE_CHECKING:
    from amltk.pipeline import Node

HP: TypeAlias = dict[str, Any]

PAIR = 2


def _parse_hp(
    node_name: str,
    *,
    hp_name: str,
    hp: tuple | list | Mapping,
    delim: str = ":",
) -> HP:
    new_hp_name = f"{node_name}{delim}{hp_name}"
    match hp:
        # If the name in a dict does not match what we see in the `space` dict, raise
        case {"name": _name_in_dict} if _name_in_dict != hp_name:
            raise ValueError(
                f'Can\'t have "name" in {hp=} as it is already given the {hp_name=}.',
            )
        # Otherwise it's a dictionary with either the same name or no name, either case
        # we give it a new name prefixed by the nodes name
        case Mapping():
            return {**hp, "name": new_hp_name}
        # Bounded int/float
        case tuple() as tup if len(tup) == PAIR:
            match tup:
                case (int() | np.integer(), int() | np.integer()):
                    x, y = tup
                    return {
                        "name": new_hp_name,
                        "type": "int",
                        "lb": int(x),
                        "ub": int(y),
                    }
                case (float() | np.floating(), float() | np.floating()):
                    x, y = tup
                    return {
                        "name": new_hp_name,
                        "type": "num",
                        "lb": float(x),
                        "ub": float(y),
                    }
                case (x, y):
                    raise ValueError(
                        f"Expected {hp_name} to have same type for lower/upper bound,"
                        f"got lower: {type(x)}, upper: {type(y)}.",
                    )
        # Bool param
        case (one, two) if isinstance(one, bool) and isinstance(two, bool):
            return {"name": hp_name, "type": "bool"}
        # Categorical param
        case list() if all(isinstance(item, str) for item in hp):
            return {"name": hp_name, "type": "cat", "categories": hp}
        # Constant value
        case (one,) if isinstance(one, int | float | str | bool):
            return {"name": hp_name, "type": "cat", "categories": [one]}
        case _:
            raise ValueError(
                f"Could not parse {hp_name} as a valid HEBO distribution.\n{hp=}",
            )

    raise ValueError(f"Could not parse {hp_name} as a valid HEBO distribution.\n{hp=}")


def _parse_space(node: Node, *, flat: bool = False, delim: str = ":") -> dict[str, HP]:
    match node.space:
        case None:
            space = {}
        case list():
            space = {hp["name"]: hp for hp in node.space}
        case Mapping():
            space = {
                name: _parse_hp(node_name=node.name, hp_name=name, hp=hp, delim=delim)
                for name, hp in node.space.items()
            }
        case _:
            raise ValueError(
                f"Can't parse {node.space=} as a HEBO space for node {node.name=}.",
            )

    for child in node.nodes:
        subspace: dict[str, HP] = _parse_space(child)
        if not flat:
            _prefix = lambda _hp_name: f"{node.name}{delim}{_hp_name}"
            subspace = {
                _prefix(hp_name): {**hp, "name": _prefix(hp_name)}
                for hp_name, hp in subspace.items()
            }

        for hp_name, hp in subspace.items():
            if hp_name in space:
                raise ValueError(
                    f"Duplicate name {hp_name} already in space from space of "
                    f"{node.name}\nCurrently parsed space: {space}",
                )
            space[hp_name] = hp

    return space


def parser(
    node: Node,
    *,
    flat: bool = False,
    conditionals: bool = False,
    delim: str = ":",
) -> DesignSpace:
    """Parse a Node and its children into a hebo DesignSpace.

    Args:
        node: The Node to parse
        flat: Whether to have a heirarchical naming scheme for nodes and their children.
        conditionals: Whether to include conditionals in the space from a
            [`Choice`][amltk.pipeline.Choice]. If this is `False`, this will
            also remove all forbidden clauses and other conditional clauses.
            The primary use of this functionality is that some optimizers do not
            support these features.

            !!! warning "Not yet supported"

                This functionality is not yet supported in HEBO

        delim: The delimiter to use for the names of the hyperparameters
    """
    if conditionals:
        raise NotImplementedError("Conditionals are not yet supported with HEBO.")

    space = _parse_space(node=node, flat=flat, delim=delim)
    hp_values = list(space.values())
    return DesignSpace().parse(hp_values)
