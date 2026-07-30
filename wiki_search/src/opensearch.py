import abc
import contextlib
import json
from collections.abc import Generator
from pathlib import Path
from typing import NotRequired, TypedDict, Unpack

import requests

from src import filesystem
from src.env import Env


class BaseBulkRequestCreator[Doc: dict](abc.ABC):
    """
    Creates OpenSearch Bulk API create requests.
    """

    @abc.abstractmethod
    def action_document_pairs(
        self,
    ) -> Generator[tuple[CreateAction, Doc]]:
        """
        Yield pairs of create actions and documents.
        """

    def create_bulk_requests(
        self, approx_size_bytes: int = 5e6
    ) -> Generator[tuple[int, str]]:
        """
        Yield request bodies of some approximate max size.

        Returns:
            Tuple with integer denoting how many documents have been
            processed thus far, and the next request body to send.
        """

        action_and_doc = "{action}\n{doc}\n"
        current_request_body = ""
        i = 0
        for action, document in self.action_document_pairs():
            i += 1
            current_request_body += action_and_doc.format(
                action=json.dumps(action), doc=json.dumps(document)
            )
            if len(current_request_body) > approx_size_bytes:
                yield i, current_request_body
                current_request_body = ""

        if len(current_request_body) > 0:
            yield i, current_request_body


class BaseREST[ResponseType](abc.ABC):
    @abc.abstractmethod
    def bulk(self, body: str) -> ResponseType:
        """
        Send bulk indexing request.
        """

    @abc.abstractmethod
    def delete_index(self, index: str) -> ResponseType: ...


class REST(BaseREST[requests.Response]):
    def __init__(self, url: str, session: requests.Session):
        self.session = session
        self.url = url

    def bulk(self, body: str):

        final_url = f"{self.url}/_bulk"
        response = self.session.post(
            final_url, data=body, headers={"Content-Type": "application/x-ndjson"}
        )
        return response

    def delete_index(self, index: str):

        final_url = f"{self.url}/{index}"
        return self.session.delete(final_url)


@contextlib.contextmanager
def default_rest():
    """
    Context manager that returns a `REST` API using default settings for the
    session and the REST URL.
    """

    try:
        s = create_default_session()
        env = Env()
        api = REST(f"https://localhost:{env.opensearch_rest_port}", s)
        yield api
    finally:
        s.close()


def create_session(*, user_pass: tuple[str, str], verify_cert: Path | bool):
    """
    Create a session suitable for connecting with a local OpenSearch REST API.

    Arguments:
        verify_cert:
            A path to a certificate file (or directory),
            or boolean indicating whether certificate should be verified
            the normal way.

            For e.g. a file path, point to the file that contains the same
            certificate held by the OpenSearch nodes under
            /usr/share/opensearch/config/root-ca.pem.
    """

    session = requests.Session()
    session.auth = user_pass
    session.verify = verify_cert
    return session


def create_default_session():
    """
    Create a default session, assuming username of 'admin' and password
    from the default environment location. `verify_cert` is set to False.
    """

    fs = filesystem.FileSystem()
    env = Env()
    return create_session(
        user_pass=("admin", env.opensearch_password), verify_cert=fs.root_ca
    )


class ActionMetadata(TypedDict):
    """
    Contents of a OpenSearch Bulk API request Action.

    See https://docs.opensearch.org/latest/api-reference/document-apis/bulk/#action-metadata-fields
    """

    _index: str
    _id: NotRequired[str]


class CreateAction(TypedDict):
    create: ActionMetadata


class ActionCreator:
    """
    OpenSearch Bulk API action creator.

    See https://docs.opensearch.org/latest/api-reference/document-apis/bulk/#actions
    """

    @staticmethod
    def create(**kwargs: Unpack[ActionMetadata]) -> CreateAction:
        return CreateAction(create=kwargs)
