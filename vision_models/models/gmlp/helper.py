from itertools import repeat
import collections.abc

from typing import Union, Tuple, Iterable, TypeVar, Callable

T = TypeVar('T', int, float)


def _to_ntuple(n) -> Callable[[Union[T, Iterable[T]]], Tuple[T, ...]]:
    """数値を長さnのtupleに変換する関数を返す"""

    def parse(x: Union[T, Iterable[T]]) -> Tuple[T, ...]:
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _to_ntuple(1)
to_2tuple = _to_ntuple(2)
to_3tuple = _to_ntuple(3)
to_4tuple = _to_ntuple(4)
to_ntuple = _to_ntuple
