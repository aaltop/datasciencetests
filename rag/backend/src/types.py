import typing as t

class Equatable(t.Protocol):
    def __eq__(self, other: t.Self) -> bool: ...

class Hashable(Equatable):
    def __hash__(self) -> int: ...

class Comparable(t.Protocol):
    def __lt__(self, other: t.Self) -> bool: ...