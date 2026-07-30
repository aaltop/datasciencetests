import tomllib as toml
from pathlib import Path

from src import filesystem


class Env:
    def __init__(self, env_file: Path | None = None):

        fs = filesystem.FileSystem()

        env_file = fs.root / "env.toml" if env_file is None else env_file

        with open(env_file, "rb") as f:
            env = toml.load(f)

        self.opensearch_password = env["opensearch_password"]
        self.opensearch_rest_port = env["opensearch_rest_port"]
