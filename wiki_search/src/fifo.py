import collections


class FIFO[T]:
    """
    First In - First Out -queue.
    """

    def __init__(self, maxlen: int = None):

        self.queue = collections.deque[T](maxlen=maxlen)

    def push(self, obj: T):

        self.queue.append(obj)

    def pop(self):

        return self.queue.popleft()

    def __len__(self):

        return len(self.queue)
