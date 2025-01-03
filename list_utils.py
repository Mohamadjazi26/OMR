"""List processing and handling utilities."""

import math
import typing as tp

T = tp.TypeVar('T')


def find_index(items: tp.List[T], value: T) -> int:
    if isinstance(value, str):
        return next(i for i, x in enumerate(items)
                    if x.strip() == value.strip())
    return next(i for i, x in enumerate(items) if x == value)


def find_greatest_value_indexes(values: tp.List[float],
                                n: int) -> tp.List[int]:
    items_copy = values.copy()
    indexes = []
    for _ in range(n):
        max_index = find_max_value_index(items_copy)
        indexes.append(max_index)
        items_copy[max_index] = -math.inf
    return indexes


def find_max_value_index(values: tp.List[float]) -> int:
    max_value = -math.inf
    max_index = -1
    for i, value in enumerate(values):
        if value > max_value:
            max_value = value
            max_index = i
    return max_index


def is_adjacent_indexes(items: tp.List[tp.Any], index_a: int, index_b: int):
    return next_index(items, index_a) == index_b or prev_index(
        items, index_a) == index_b


def unnest(nested: tp.List[tp.List[tp.List[tp.Any]]]):
    return [e[0] for e in nested]


def call_on_some(items: tp.List[tp.Any], indexes: tp.List[int],
                 fn: tp.Callable[[tp.Any], tp.Any]) -> list:
    return [el if i not in indexes else fn(el) for i, el in enumerate(items)]


def next_index(items: tp.List[tp.Any], index: int) -> int:
    return index + 1 if index + 1 < len(items) else 0


def prev_index(items: tp.List[tp.Any], index: int) -> int:
    return index - 1 if index - 1 >= 0 else len(items) - 1


def continue_index(items: tp.List[tp.Any], index_a: int, index_b: int) -> int:
    if index_b >= index_a:
        return next_index(items, index_b)
    return prev_index(items, index_b)


Pair = tp.Tuple[int, int]

def arrange_like_rays(pair_a: Pair, pair_b: Pair) -> tuple:
    shared = next((a for a, b in zip(pair_a, pair_b) if a == b), None)
    if shared is not None:
        pair_a_ = pair_a if pair_a[1] == shared else tuple(reversed(pair_a))
        pair_b_ = pair_b if pair_b[0] == shared else tuple(reversed(pair_b))
        return pair_a_, pair_b_
    else:
        return pair_a, pair_b


def arrange_index_to_first(items: tp.List[tp.Any], index: int) -> list:
    if index >= len(items) or index < 0:
        raise IndexError("Index is invalid.")

    result = [items[index]]
    i = next_index(items, index)
    while i != index:
        result.append(items[i])
        i = next_index(items, i)
    return result


def determine_which_is_next(items: tp.List[tp.Any], index_a: int,
                            index_b: int) -> int:
    if next_index(items, index_a) == index_b:
        return index_b
    return index_a


def strip_all(items: tp.List[str]) -> tp.List[str]:
    return [item.strip() for item in items]


def remove_index(items: tp.List[T], index: int) -> tp.List[T]:
    return [item for i, item in enumerate(items) if i != index]


def transpose(matrix: tp.List[tp.List[T]]) -> tp.List[tp.List[T]]:
    if len(set([len(row) for row in matrix])) != 1:
        raise ValueError(
            "Input matrix rows must all have the same length for transposing.")
    return [[row[col_index] for row in matrix]
            for col_index in range(len(matrix[0]))]


def count_trailing_empty_elements(items: tp.List[tp.Any]) -> int:
    return next((i for i, x in enumerate(reversed(items)) if x != ""), len(items))