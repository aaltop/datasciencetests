import tomllib as toml
from pathlib import Path


class Env:
    def __init__(self, env_file: Path):

        with open(env_file, "rb") as f:
            env = toml.load(f)

        self.opensearch_password = env["opensearch_password"]
        self.opensearch_rest_port = env["opensearch_rest_port"]
