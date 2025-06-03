from dotenv import load_dotenv
import mlflow

import subprocess
import os


load_dotenv()
PORT = os.environ["MLFLOW_PORT"]

try:
    mlflow.set_tracking_uri(f"http://localhost:{PORT}")
    command = f"mlflow -- gc".split()
    subprocess.run(
        command,
    )
except KeyboardInterrupt:
    pass