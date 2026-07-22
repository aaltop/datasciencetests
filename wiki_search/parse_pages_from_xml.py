from io import TextIOBase
import json
import os
from pathlib import Path
import re
import time
from typing import TypedDict
import xml.etree.ElementTree as ET
import multiprocessing as mp
from multiprocessing.pool import AsyncResult, Pool

from src.fifo import FIFO
from src.ansi import ANSI


class FileSystem:
    root = Path()

    data_dir = root / "data"

    parsed_pages_dir = data_dir / "parsed_pages"

    wiki_xml = data_dir / "wiki.xml"

    def __init__(self):

        dirs_to_make = [self.parsed_pages_dir]
        for d in dirs_to_make:
            d.mkdir(exist_ok=True)


fs = FileSystem()


class PageDetails(TypedDict):
    """
    A Parsed page from a wikipedia XML dump.
    """

    start_line: int
    end_line: int
    text: str


def get_lines(file: Path):
    """
    Get the number of lines in `file`.
    """

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
    """
    A page from a Wikipedia XML dump.
    """

    start_line: int
    end_line: int
    page_elem: ET.Element


class PagesParser:
    tag_prefix = "{http://www.mediawiki.org/xml/export-0.11/}"

    def __init__(
        self,
        batch_size: int = None,
        xml_file: Path = None,
        forbidden_title_content: list[str] = None,
        forbidden_text_content: list[str] = None,
    ):
        """
        Arguments:
            batch_size:
                How many pages to find before sending to be processed in
                by a second process (multiprocessing).

            xml_file:
                File to read wiki content from.

            forbidden_title_content:
                Partial strings that should not be present in the titles
                of pages, e.g. ['Conversation:'] to not include any pages
                that have 'Conversation:' in the title. Set to None to
                allow all titles. Can be in regex syntax.

            forbidden_text_content:
                Partial strings that should not be present in the beginning
                of the textual body of a page. Set to None to allow all.
        """

        self._batch_size = batch_size or 10_000
        self._xml_file = xml_file or fs.wiki_xml

        self._all_title_valid = forbidden_title_content is None
        if not self._all_title_valid:
            self._forbidden_title_regex = re.compile("|".join(forbidden_title_content))

        self._all_text_valid = forbidden_text_content is None
        if not self._all_text_valid:
            self._forbidden_text_regex = re.compile("|".join(forbidden_text_content))

    def _valid_title(self, title: str):

        return self._all_title_valid or (
            self._forbidden_title_regex.search(title) is None
        )

    def _valid_text(self, text: str):

        return self._all_text_valid or (
            # only check the beginning of the string, don't want
            # to spend too much time here
            self._forbidden_text_regex.match(text) is None
        )

    def _find_next_page(
        self, f: TextIOBase, parser: ET.XMLPullParser, last_read_line: int
    ):
        """
        Arguments:
            parser:
                Needs to have events at least as ["start", "end"]
        """

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

    def parse_pages(self, num: int = None, continue_previous: bool = True):
        """
        Parse the pages from the wiki xml file `self.xml_file`.

        Arguments:
            num:
                How many pages to try to find. If None, process the entire
                xml file.

        Returns:
            The last line read from the xml file.
        """

        parser = ET.XMLPullParser(["start", "end"])
        current_line = get_last_read_line() if continue_previous else 0
        if current_line > 0:
            with open(self._xml_file, "r") as f:
                # give it the namespace info when it's not reading from the start
                # if we don't do this, it won't find the tags (when beginning reading
                # from somewhere in the middle) as it doesn't know
                # what the namespace is supposed to be (basically, using the tag_prefix
                # isn't going to work if this isn't done)
                #
                # obviously assumes that the first line contains the
                # relevant info
                parser.feed(f.readline())

        pages_to_find = num if num is not None else float("infinity")
        pages_found = 0
        pages: list[tuple[str, PageDetails]] = []
        clear_move_to_start = ANSI().erase_in_line("entire_line").cursor_to_column()
        start_time = time.time()
        processing_results = FIFO[AsyncResult]()
        processor = PageProcessor(self.tag_prefix)

        process_worker_count = os.process_cpu_count() or 1
        with mp.Pool(processes=process_worker_count, maxtasksperchild=1) as pool:
            with open(self._xml_file) as f:
                navigate_to_line(f, current_line + 1)

                while pages_found < pages_to_find:
                    raw_page = self._find_next_page(f, parser, current_line)
                    if raw_page is None:
                        break
                    current_line = raw_page["end_line"]

                    page_title, page = processor.raw_page_to_page_details(raw_page)
                    # clear the raw element after to preserve memory
                    # (might require a more thorough clear of the parser
                    # with larger xml files).
                    raw_page["page_elem"].clear()
                    # check for unwanted titles (e.g. conversations)
                    if not self._valid_title(page_title) or not self._valid_text(
                        page["text"]
                    ):
                        continue

                    pages_found += 1

                    # periodic report
                    if pages_found % 1000 == 0:
                        print(
                            "{0}Current line: {1} page count: {2} elapsed time (s): {3}".format(
                                str(clear_move_to_start),
                                page["end_line"],
                                pages_found,
                                time.time() - start_time,
                            ),
                            end="",
                            flush=True,
                        )

                    pages.append((page_title, page))

                    # process in a worker periodically
                    if len(pages) >= self._batch_size:
                        # copy the list so the clear below doesn't affect
                        # the processing
                        pages_to_process = pages[:]
                        result = self._process_pages(pages_to_process, pool)
                        processing_results.push(result)
                        pages.clear()

                        # ensure both that results list doesn't grow to
                        # too large sizes, and that there actually are
                        # workers ready to process the data so that the
                        # data queue doesn't grow too much either
                        if len(processing_results) >= process_worker_count:
                            processing_results.pop().get(10)

        if len(pages) > 0:
            processor = PageProcessor(self.tag_prefix)
            processor.write_parsed_pages(pages)

        return current_line

    def _process_pages(self, pages: list[tuple[str, PageDetails]], pool: Pool):

        processor = PageProcessor(self.tag_prefix)
        # copy the list so the clear below doesn't affect
        # the processing
        result = pool.apply_async(
            processor.write_parsed_pages,
            args=[pages],
        )
        return result


def get_last_read_line():

    file = fs.data_dir / "last_read_line.txt"
    if not file.exists():
        return 0
    return int((fs.data_dir / "last_read_line.txt").read_text())


class PageProcessor:
    def __init__(self, tag_prefix: str):
        """
        `tag_prefix` is the namespace prefix
        of the tags of the XML elements of the XML document from which
        the pages to be processed originated from.
        """

        self.tag_prefix = tag_prefix

    def raw_page_to_page_details(self, raw: RawPage) -> tuple[str, PageDetails]:
        """
        Transform the RawPage to PageDetails. `tag_prefix` is the namespace prefix
        of the tags of the XML elements of the XML document from which `raw`
        originated from.
        """

        tags = "title", "text"
        tags_and_elems = {
            tag: raw["page_elem"].find(f".//{self.tag_prefix}{tag}") for tag in tags
        }

        title = tags_and_elems["title"].text
        # maybe not the smartest choice, but works, and not that
        # many without text anyway, I don't think
        page_text = tags_and_elems["text"].text or ""
        details = PageDetails(
            start_line=raw["start_line"], end_line=raw["end_line"], text=page_text
        )
        return title, details

    def write_parsed_pages(self, pages: list[tuple[str, PageDetails]]):
        """
        Write the pages to file, returning the start and end lines that delimit
        the pages in the original XML document.
        """

        pages_dicts = dict(pages)
        start_line = pages[0][-1]["start_line"]
        end_line = pages[-1][-1]["end_line"]

        file_name = "{0}_{1}.json".format(start_line, end_line)

        with open(
            fs.parsed_pages_dir / file_name,
            "w",
        ) as f:
            json.dump(pages_dicts, f, indent=2)
        return (start_line, end_line)

    def transform_and_write_parsed_pages(self, pages: list[RawPage]):
        """
        Combines `raw_page_to_page_details()` and `write_parsed_pages()`.
        """

        return self.write_parsed_pages(
            [self.raw_page_to_page_details(pag) for pag in pages]
        )


def get_forbidden_content(file: Path):
    """
    Read forbidden content from `file`. Each line is expected
    to have one forbidden example.
    """

    forbidden: list[str] = []

    with open(file, "r") as f:
        forbidden = list(map(lambda x: x.strip(), f.readlines()))

    return forbidden


def main():

    parser = PagesParser(
        forbidden_title_content=get_forbidden_content(
            Path() / "forbidden_title_content.txt"
        ),
        forbidden_text_content=get_forbidden_content(
            Path() / "forbidden_text_content.txt"
        ),
    )

    last_read_line = parser.parse_pages(continue_previous=False)

    with open(fs.data_dir / "last_read_line.txt", mode="w") as f:
        f.write(str(last_read_line))


if __name__ == "__main__":
    start = time.time()
    main()
    print("\nTotal run time (s):", time.time() - start)
