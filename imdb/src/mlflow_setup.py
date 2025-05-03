import mlflow
from dotenv import load_dotenv

import os
import logging

logger = logging.getLogger(__name__)

def mlflow_setup(request_timeout = 5, max_retries = 2):
    '''
    Set the tracking URI for mlflow, and set HTTP request settings.
    '''
    load_dotenv()
    uri = f"http://127.0.0.1:{os.environ['MLFLOW_PORT']}"
    logger.info("Setting mlflow uri: %s", uri)
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(request_timeout)
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(max_retries)
    mlflow.set_tracking_uri(f"http://127.0.0.1:{os.environ['MLFLOW_PORT']}")