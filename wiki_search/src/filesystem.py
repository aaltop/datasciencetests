from pathlib import Path


class FileSystem:
    """
    Contains paths to various relevant locations on the local filesystem.
    Will create some directories at initialisation if the directories don't
    exist yet.
    """

    root = Path()

    data_dir = root / "data"

    parsed_pages_dir = data_dir / "parsed_pages"
    page_embeddings_dir = data_dir / "page_embeddings"
    wiki_xml = data_dir / "wiki.xml"

    certs_dir = root / "certs"
    root_ca = certs_dir / "root_ca.pem"

    def __init__(self):

        dirs_to_make = [self.data_dir, self.parsed_pages_dir, self.page_embeddings_dir]
        for d in dirs_to_make:
            d.mkdir(exist_ok=True)

    def parsed_pages_file(self, first_read_line: int, last_read_line: int):
        """
        Create the file name for a parsed pages file.

        ## Arguments:
            first_read_line:
                The line on which the first page out of the set of pages
                started in the original XML file.
            last_read_line:
                The line on which the last page out of the set of pages
                ended in the original XML file.
        """

        return self.parsed_pages_dir / f"{first_read_line}_{last_read_line}.json"

    def page_embeddings_file(self, parsed_pages_file: Path):
        """
        Create the file name for a page embeddings file.

        ## Arguments:
            parsed_pages_file:
                See `self.parsed_pages_file()`.
        """

        return self.page_embeddings_dir / f"{parsed_pages_file.stem}.pt"

    def get_parsed_pages_files(self):
        """
        Returns:
            The parsed pages files from `self.parsed_pages_dir`, sorted
            in to the order from which they would have been read from
            the original XML file.
        """

        return sorted(
            self.parsed_pages_dir.iterdir(),
            key=lambda x: int(str(x.stem).split("_")[0]),
        )
