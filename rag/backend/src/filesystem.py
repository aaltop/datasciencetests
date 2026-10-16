from pathlib import Path
import abc

class FileSystem:

    root = Path()
    data = root / "data"
    text_data = data / "python-3.15-docs-text"
    output = root / "output"
    encodings_output = output / "encodings"

fs = FileSystem()