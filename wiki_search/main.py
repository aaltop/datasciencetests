from io import TextIOBase
import itertools
import json
from pathlib import Path
import time
from typing import TypedDict
import xml.etree.ElementTree as ET

from src.ansi import ANSI


class FileSystem:
    root = Path()

    data_dir = root / "data"

    wiki_xml = data_dir / "wiki.xml"


fs = FileSystem()


class PageDetails(TypedDict):
    start_line: int
    end_line: int
    text: str


def get_lines(file: Path):

    lines = 0
    with open(fs.wiki_xml) as f:
        while f.readline() != "":
            lines += 1
    return lines


def navigate_to_line(file_handle: TextIOBase, line: int):
    """
    Navigate up to the `line`-nth line such that the next read line
    from `file_handle` would the the `line`-nth line.
    """

    for i in range(line - 1):
        file_handle.readline()


class RawPage(TypedDict):
    start_line: int
    end_line: int
    page_elem: ET.Element


class PagesParser:
    tag_prefix = "{http://www.mediawiki.org/xml/export-0.11/}"

    def _find_next_page(
        self, f: TextIOBase, parser: ET.XMLPullParser, last_read_line: int
    ):

        current_line = last_read_line
        page = RawPage()
        while True:
            line = f.readline()
            if line == "":
                return None
            current_line += 1
            parser.feed(line)
            for event, elem in parser.read_events():
                if elem.tag == self.tag_prefix + "page":
                    if event == "start":
                        page["start_line"] = current_line
                    if event == "end":
                        page["end_line"] = current_line
                        page["page_elem"] = elem
                        # assuming that there aren't multiple page elements on one line
                        return page

    def find_pages(self, num: int, continue_previous: bool = True):

        parser = ET.XMLPullParser(["start", "end"])
        current_line = get_last_read_line() if continue_previous else 0
        if current_line > 0:
            with open(fs.wiki_xml, "r") as f:
                # give it the namespace info when it's not reading from the start
                # if we don't do this, it won't find the tags (when beginning reading
                # from somewhere in the middle) as it doesn't know
                # what the namespace is supposed to be (basically, using the tag_prefix
                # isn't going to work if this isn't done)
                parser.feed(f.readline())
        pages_to_find = num
        pages_found = 0
        pages: list[RawPage] = []
        clear_move_to_start = ANSI().erase_in_line("entire_line").cursor_to_column()
        start = time.time()
        with open(fs.wiki_xml) as f:
            navigate_to_line(f, current_line + 1)
            while pages_found < pages_to_find:
                page = self._find_next_page(f, parser, current_line)
                if page is None:
                    break
                pages_found += 1
                if pages_found % 1000 == 0:
                    print(
                        "{0}Current line: {1} page count: {2} elapsed time (s): {3}".format(
                            str(clear_move_to_start),
                            page["end_line"],
                            pages_found,
                            time.time() - start,
                        ),
                        end="",
                        flush=True,
                    )
                current_line = page["end_line"]
                pages.append(page)

        return [self._raw_page_to_page_details(page) for page in pages]

    def _raw_page_to_page_details(self, raw: RawPage) -> tuple[str, PageDetails]:

        tags = "title", "text"
        tags_and_elems = {
            tag: raw["page_elem"].findall(f".//{self.tag_prefix}{tag}") for tag in tags
        }

        title = tags_and_elems["title"][0].text
        page_text = tags_and_elems["text"][0].text
        details = PageDetails(
            start_line=raw["start_line"], end_line=raw["end_line"], text=page_text
        )
        return title, details


def get_last_read_line():

    file = fs.data_dir / "last_read_line.txt"
    if not file.exists():
        return 0
    return int((fs.data_dir / "last_read_line.txt").read_text())


def main():

    parser = PagesParser()

    pages = parser.find_pages(100000, False)
    last_read_line = pages[-1][-1]["end_line"]

    with open(fs.data_dir / "last_read_line.txt", mode="w") as f:
        f.write(str(last_read_line))

    for batch in itertools.batched(pages, 10000):
        pages_dicts = dict(batch)
        batch_first_read_line = batch[0][-1]["start_line"]
        batch_last_read_line = batch[-1][-1]["end_line"]

        file_name = "parsed_pages_{0}_{1}.json".format(
            batch_first_read_line, batch_last_read_line
        )

        with open(
            fs.data_dir / file_name,
            "w",
        ) as f:
            json.dump(pages_dicts, f, indent=2)


if __name__ == "__main__":
    main()
