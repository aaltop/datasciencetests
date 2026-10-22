"""
itertools
"""

from collections.abc import Iterable, Callable
from collections import defaultdict

from src.types import Hashable

# the itertools.groupby requires that the values be sorted first for
# the thing to work as expected (which is to group all values that belong
# to the same group together). This function allows not having to remember to
# sort and in fact to speed up the process by not having to sort at all.
#
# the stdlib version probably does it the way it does because it actually
# then benefits from the iter paradigm, as it doesn't have to calculate
# all the values all at once.
def groupby[T, H: Hashable](iterable: Iterable[T], key: Callable[[T], H]) -> list[tuple[H, list[T]]]:
    '''
    Group values in `iterable` based on each value's output from `key`,
    where outputs that hash equal are considered to be in the same group.
    '''

    groups = defaultdict(list)
    for val in iterable:
        groups[key(val)].append(val)

    return list(groups.items())