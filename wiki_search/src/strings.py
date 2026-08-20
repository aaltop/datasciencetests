import functools
import re


class Section:
    """
    Represents a section (sub-string) of a string.
    """

    def __init__(
        self,
        delimited: bool,
        start: int,
        end: int,
        content: str,
        left_delimiter: str,
        right_delimiter: str,
    ):

        self.delimited = delimited
        self.start = start
        self.end = end
        self.content = content
        "All content, including possible delimiters"
        self.contained_content = self.content.removeprefix(left_delimiter).removesuffix(
            right_delimiter
        )
        "The content between the delimiters. Equals `self.content` if not delimited."

    @staticmethod
    def section_factory(left_delimiter: str, right_delimiter: str):

        return functools.partial(
            Section, left_delimiter=left_delimiter, right_delimiter=right_delimiter
        )


class NestedDelimitedFinder:
    """
    Find all delimited areas in a string.
    """

    def __init__(
        self,
        left_re: str,
        right_re: str,
        left: str | None = None,
        right: str | None = None,
    ):

        if left is None:
            left = left_re
        self.left = left
        if right is None:
            right = right_re
        self.right = right

        self._section_factory = Section.section_factory(self.left, self.right)

        self.re = re.compile(rf"{left_re}|{right_re}")

    def find(self, content: str):

        order = sorted(
            [
                (match.start(), match.end(), match.group())
                for match in self.re.finditer(content)
            ]
        )

        # The depth of the current delimited area.
        depth = 0
        start = -1
        idx = 0
        sections: list[Section] = []
        for spot in order:
            if spot[-1] == self.left:
                if depth == 0:
                    # up to this left delimiter, there is a non-delimited
                    # section
                    end = spot[0]
                    sections.append(
                        self._section_factory(
                            delimited=False,
                            start=idx,
                            end=end,
                            content=content[idx:end],
                        )
                    )
                depth += 1
                if depth < 2:
                    start = spot[0]
            else:
                if depth < 1:
                    # encountering right delimiter before a corresponding
                    # left one has been encountered, ignore
                    continue
                if depth < 2:
                    # new non-delimited section starts here
                    idx = spot[1]
                    # only if within one set of delimiter (depth 1) is the right
                    # delimiter the correct one to correspond the
                    # originally found left delimiter
                    sections.append(
                        self._section_factory(
                            delimited=True,
                            start=start,
                            end=idx,
                            content=content[start:idx],
                        )
                    )
                depth -= 1

        # include any remaining, non-delimited content from the end
        # if it exists
        if len(content) > idx:
            sections.append(
                self._section_factory(
                    delimited=False,
                    start=idx,
                    end=len(content),
                    content=content[idx : len(content)],
                )
            )

        return sections
