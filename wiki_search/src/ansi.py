from typing import Literal


CSI = "\x1b["


class ANSI:
    """
    Class for working with ANSI escape codes (e.g. moving cursor in terminal,
    changing terminal text front/back color). Commands can be chain-created,
    while passing the instance to str() creates the code, and self.clear()
    clears the command list.
    """

    def __init__(self):

        self._commands: list[str] = []

    def erase_in_line(
        self,
        how: Literal["cursor_to_end", "cursor_to_begin", "entire_line"],
    ):
        """
        Erase characters from the current line, depending on `how`.
        """

        code = dict(cursor_to_end="1", cursor_to_begin="2", entire_line="3")[how]
        self._commands.append(CSI + code + "K")
        return self

    def cursor_to_column(self, col: int = 0):
        """
        Move the cursor to the specified column (i.e. character position)
        on the current line, with 0 meaning first column.
        """

        # first col is 1 for ANSI code, while using 0 makes more sense for
        # python
        actual_col = str(col + 1)
        self._commands.append(CSI + actual_col + "G")
        return self

    def __str__(self):

        return " ".join(self._commands)

    def clear(self):
        self._commands.clear()
