from typing import Literal

class ANSI:

    CODE = "\x1b["

    def cursor_previous_line(self, n:int = 1):
        return f"{ANSI.CODE}{n}F"

    def cursor_next_line(self, n:int = 1):
        return f"{ANSI.CODE}{n}E"


    def erase_in_display(self, how: Literal["cursor_to_end", "cursor_to_begin", "all"]="cursor_to_end"):
        code = -1
        match how:
            case "all":
                code = 2
            case "cursor_to_begin":
                code = 1
            case "cursor_to_end":
                code = 0

        return f"{ANSI.CODE}{code}J"

class TerminalPrinter:
    '''
    Printer for printing to terminal, allowing easy erasure of previously
    printed content.
    '''

    def __init__(self):

        self.newlines = 0
        self.ansi = ANSI()

    def write(self, s: str):

        self.newlines += len(s.split("\n")) - 1
        print(s, end="", sep="")

    def clear(self):
        '''
        return to starting line and clear anything written thus far.
        '''
        print(self.ansi.cursor_previous_line(self.newlines) + self.ansi.erase_in_display(), end="")
        self.newlines = 0

    def print(
        self,
        *values: object,
        sep: str | None = " ",
        end: str | None = "\n",
        flush: bool = False
    ):
        print(*values, sep=sep, end=end, file=self)
        if flush:
            print("", sep="", end="", flush=True)