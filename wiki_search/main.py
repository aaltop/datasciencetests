from io import TextIOBase
import json
from pathlib import Path
from typing import TypedDict
import xml.etree.ElementTree as ET


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

    def find_next_page(
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
                # what the namespace is supposed to be
                parser.feed(f.readline())
        pages_to_find = num
        pages_found = 0
        pages: list[RawPage] = []
        with open(fs.wiki_xml) as f:
            navigate_to_line(f, current_line + 1)
            while pages_found < pages_to_find:
                page = self.find_next_page(f, parser, current_line)
                if page is None:
                    break
                pages_found += 1
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


# def find_pages():

#     tag_prefix = "{http://www.mediawiki.org/xml/export-0.11/}"

#     parser = ET.XMLPullParser(["start", "end"])
#     last_read_line = get_last_read_line()
#     with open(fs.wiki_xml) as f:
#         current_line = 0
#         page_elem: ET.Element | None = None
#         pages_to_find = 10
#         pages_found = 0
#         # start, stop line, the element itself
#         pages: list[list[int, int, ET.Element]] = []
#         while pages_found < pages_to_find:
#             line = f.readline()
#             if line == "":
#                 break
#             current_line += 1
#             parser.feed(line)
#             for event, elem in parser.read_events():
#                 if elem.tag == tag_prefix + "page":
#                     if event == "start":
#                         pages.append([])
#                         pages[-1].append(current_line)
#                     if event == "end":
#                         pages[-1].append(current_line)
#                         pages[-1].append(elem)
#                         pages_found += 1


def get_last_read_line():

    file = fs.data_dir / "last_read_line.txt"
    if not file.exists():
        return 0
    return int((fs.data_dir / "last_read_line.txt").read_text())


def main():

    # tag_prefix = "{http://www.mediawiki.org/xml/export-0.11/}"

    # parser = ET.XMLPullParser(["start", "end"])
    # last_read_line = get_last_read_line()
    # with open(fs.wiki_xml) as f:
    #     current_line = 0
    #     page_elem: ET.Element | None = None
    #     pages_to_find = 10
    #     pages_found = 0
    #     # start, stop line, the element itself
    #     pages: list[list[int, int, ET.Element]] = []
    #     while pages_found < pages_to_find:
    #         line = f.readline()
    #         if line == "":
    #             break
    #         current_line += 1
    #         parser.feed(line)
    #         for event, elem in parser.read_events():
    #             if elem.tag == tag_prefix + "page":
    #                 if event == "start":
    #                     pages.append([])
    #                     pages[-1].append(current_line)
    #                 if event == "end":
    #                     pages[-1].append(current_line)
    #                     pages[-1].append(elem)
    #                     pages_found += 1

    # pages_dicts: dict[str, PageDetails] = dict()
    # last_read_line = 0
    # for start_line, end_line, page_elem in pages:
    #     last_read_line = end_line
    #     print(f"{start_line=}, {end_line=}")
    #     tags = "title", "text"
    #     if page_elem is not None:
    #         tags_and_elems = {
    #             tag: page_elem.findall(f".//{tag_prefix}{tag}") for tag in tags
    #         }

    #         title = tags_and_elems["title"][0].text
    #         page_text = tags_and_elems["text"][0].text
    #         details = PageDetails(
    #             start_line=start_line, end_line=end_line, text=page_text
    #         )
    #         pages_dicts[title] = details

    parser = PagesParser()

    pages = parser.find_pages(3, True)
    last_read_line = pages[-1][-1]["end_line"]
    pages_dicts = dict(pages)

    with open(fs.data_dir / "last_read_line.txt", mode="w") as f:
        f.write(str(last_read_line))

    with open(fs.data_dir / "parsed_pages.json", "w") as f:
        json.dump(pages_dicts, f, indent=2)


if __name__ == "__main__":
    main()

    # with open(fs.wiki_xml, "r") as f:
    #     navigate_to_line(f, 562 + 106)
    #     for i in range(3):
    #         print(f.readline(), end="")
