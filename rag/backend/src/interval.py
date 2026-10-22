from src.types import Comparable, Equatable
from src.itools import groupby

type Interval[T: Comparable] = tuple[T, T]

def union[T: Comparable](first: Interval[T], second: Interval[T]) -> Interval[T] | None:
    '''
    Return a new tuple that is the combination of the passed ones
    if they overlap, otherwise return None.
    '''
    if (first[0] >= second[0]) and (first[0] <= second[1]):
        # first starts between second's ends
        return (
            second[0],
            max(second[1], first[1])
        )
    elif (second[0] >= first[0]) and (second[0] <= first[1]):
        # second starts between first's ends
        return (
            first[0],
            max(second[1], first[1])
        )
    else:
        return None

class ScoredGroupInterval[T: Comparable, K: Equatable]:

    def __init__(self, score: float, group: K, interval: Interval[T]):

        self._score = score
        self._group = group
        self._interval = interval

    @property
    def score(self) -> float:
        return self._score

    @property
    def group(self) -> K:
        return self._group

    @property
    def interval(self) -> Interval[T]:
        return self._interval

    def __str__(self):

        return f"ScoredGroupInterval(score={self.score}, group={self.group}, interval={self.interval})"

    def __repr__(self):
        return self.__str__()

def groupby_union[T: ScoredGroupInterval](vals: list[T]) -> list[T]:
    '''
    Group the passed intervals and create unions of each pair of intervals
    that overlap in each group, returning ultimately a flat (non-grouped) list.
    Retains a score based on highest score in each union (good for sorting).
    '''
    
    groups = groupby(vals, key=lambda x: x.group)

    def rotate(l: list):
        first = l[0]
        l[:-1] = l[1:]
        l[-1] = first

    joined: list[T] = []
    for _, group in groups:
        group_start_length = len(group)
        if group_start_length < 2:
            joined.extend(group)
            continue
        rotation_count = 0
        # 1. for a list of values, take first and compare against all others.
        # For all that overlap, combine and replace the chosen
        # first with the combined (any that would have overlapped with
        # the first or any overlaps of that would now overlap with the
        # combined one as well). Leave a null marker (such as None)
        # for every consumed value.
        # 2. once the end of the list is reached, continue from the next element
        # after rotating indexes i -> i-1 with wrap-around, if the number
        # of rotations performed thus far is less than the length of
        # the list minus one; otherwise stop.
        # (it might be some of the values that were consumed in the
        # previous combination would've overlapped with the current
        # element, which would otherwise not be tested).
        # 3. If the current element is a null marker, rotate again
        # as in 2. Otherwise, continue from 1.
        while True:
            first = group[0]
            if first is not None:
                for i in range(1, group_start_length):
                    other = group[i]
                    if other is None:
                        continue
                    overlap = union(first.interval, other.interval)
                    if overlap is None:
                        pass
                    else:
                        first: T = ScoredGroupInterval(max(first.score, other.score), first.group, overlap)
                        group[0] = first
                        group[i] = None

            if rotation_count < group_start_length-1:
                rotation_count += 1
                rotate(group)
            else:
                joined.extend(filter(None, group))
                break
                
    return joined