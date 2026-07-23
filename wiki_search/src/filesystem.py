from pathlib import Path


class FileSystem:
    """
    Contains paths to various relevant locations on the local filesystem.
    """

    root = Path()

    data_dir = root / "data"

    parsed_pages_dir = data_dir / "parsed_pages"

    wiki_xml = data_dir / "wiki.xml"

    def __init__(self):

        dirs_to_make = [self.parsed_pages_dir]
        for d in dirs_to_make:
            d.mkdir(exist_ok=True)
