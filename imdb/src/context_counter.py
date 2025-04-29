import collections
from csv import DictWriter
from io import StringIO

class ContextCounter(dict):
    '''
    For counting the context (e.g. positive, negative) that words appear in.
    '''

    def __init__(self):
        '''
        For counting the context that words appear in.
        '''

        self._contexts = set()

    @property
    def contexts(self):

        return sorted(self._contexts)

    def __missing__(self, key):

        self[key] = collections.Counter()
        return self[key]


    def add(self, context, words:list):
        '''
        Add one to the counter of `context` for each word in `words`.
        '''

        self._contexts.add(context)

        for word in words:

            self[word].update([context])

    def add_each(self, contexts:list, words:list):
        '''
        Add each pair of values. Vectorised (convenience) version of `add()`.
        '''

        for c, w in zip(contexts, words):
            self.add(c,w)

    def to_table(self, sort_func=None) -> dict:
        '''
        Create a flattened table-like structure of the contents.
        Also includes the total count for each item.

        Supply <sort_func> to sort items. <sort_func> is used as
        the key to `sorted()`.
        '''

        base_row = (
            { "word": None } 
            | {cont: 0 for cont in self.contexts}
            | {"total": 0}
        )


        table = []
        for key, value in self.items():

            new_row = {"word": key} | value | {"total": value.total()}
            table.append(base_row | new_row)

        if not (sort_func is None):

            table = sorted(table, key=sort_func, reverse=True)

        return table

    def to_csv(self):
        '''
        Write data to csv. Sorts by total count first, word second,
        in descending order.
        '''

        stringio = StringIO()
        writer = DictWriter(
            stringio,
            ["word"] + self.contexts + ["total"],
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            self.to_table(
                sort_func=lambda val: (val["total"], val["word"])
            )
        )
        return stringio.getvalue()
    

    def __str__(self):

        return self.to_csv()

            



if __name__ == "__main__":

    context_counter = ContextCounter()
    context_counter.add("negative", ["bacon", "foo", "bar"])
    context_counter.add("positive", ["bacon", "bar"])
    print(context_counter)
    with open("test.csv", "w") as f:
        f.write(context_counter.to_csv())