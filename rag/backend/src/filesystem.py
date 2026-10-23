from pathlib import Path
from collections.abc import Iterable
from itertools import zip_longest
from typing import Self

class FileSystem:

    root = Path()
    data = root / "data"
    text_data = data / "python-3.15-docs-text"
    output = root / "output"
    encodings_output = output / "encodings"

fs = FileSystem()

def dir_and_file(file_names: Iterable[Path]) -> tuple[list[Path], list[Path]]:

    dirs, files = [], []
    for file in file_names:
        if file.is_file():
            files.append(file)
        else:
            # bit of an assumption
            dirs.append(file)

    return (dirs, files)

def child_depth(child: Path, parent: Path):
    '''
    Positive values indicate depth of `child` relative to `parent`,
    with 0 indicating the two are the same, and -1 indicating no
    child-parent dependence.
    '''


    child_parts = child.parts
    parent_parts = parent.parts
    if len(child_parts) < len(parent_parts):
        return -1

    sames = [c == p for c, p in zip_longest(child.parts, parent.parts, fillvalue="")]
    sames_count = sum(sames)

    return max(-1, len(child_parts) - sames_count)
    

class Directory:


    def __init__(self, path: Path):

        self.path = path
        self._dirs: dict[Path, Self] = dict()
        self._files: set[Path] = set()

    def __contains__(self, path: Path):
        return self.contains_dir_path(path) or self.contains_file(path)

    def contains_dir_path(self, path: Path):
        return path in self._dirs
    
    def contains_dir(self, _dir: Directory):
        return self.contains_dir_path(_dir.path)

    
    def contains_file(self, path: Path):
        return path in self._files

    def add_dirs(self, *dirs: Directory):
        self._dirs.update((d.path, d) for d in filter(lambda x: not self.contains_dir(x), dirs))
        return self

    def add_dir_paths(self, *dirs: Path):
        self.add_dirs(*(Directory(d) for d in dirs))

    def add_files(self, *paths: Path):
        self._files.update(paths)
        return self

    def add_paths(self, *paths: Path):
        for path in paths:
            if path.is_dir():
                self.add_dir_paths(path)
            else:
                self.add_files(path)
        return self

    @property
    def dirs(self):
        return list(self._dirs.values())

    @property
    def files(self):
        return list(self._files)

    def __eq__(self, other: Self):
        return self.path == other.path

    def __hash__(self):
        return self.path

    def get_dir(self, path: Path):
        return self._dirs.get(path)

    def show(self):
        buf = f"{str(self.path)}\n"
        paths = ["|—"+str(d.path)+"/" for d in self.dirs] + ["|—"+str(f) for f in self.files]
        return buf+"\n".join(sorted(paths))
    
    def __str__(self):
        return self.show()

    def __repr__(self):
        return f"Directory({str(self.path)})"


def flat_to_nested(file_name_parts: list[Iterable[str]]):


    root_dirs: dict[Path, Directory] = dict()
    for parts in file_name_parts:
        curr_path, *rest = parts
        curr_path = Path(curr_path)
        if curr_path not in root_dirs:
            root_dirs[curr_path] = Directory(curr_path)
        curr_dir = root_dirs[curr_path]
        while True:
            if len(rest) == 0:
                break
            first, *rest = rest
            curr_path = curr_path / first
            curr_dir.add_paths(curr_path)
            new_curr = curr_dir.get_dir(curr_path)
            curr_dir = new_curr if new_curr is not None else curr_dir

    return root_dirs