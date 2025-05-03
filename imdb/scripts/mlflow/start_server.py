from dotenv import load_dotenv

import subprocess
import os


load_dotenv()
PORT = os.environ["MLFLOW_PORT"]

try:
    command = f"mlflow -- server --host 127.0.0.1 --port {PORT}".split()
    subprocess.run(
        command,
    )
except KeyboardInterrupt:
    pass