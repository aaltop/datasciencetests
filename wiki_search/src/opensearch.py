from typing import Iterable, NotRequired, TypedDict, Unpack
from collections.abc import Generator
import abc
import json

import requests


class BaseBulkRequestCreator[RawDoc: dict, Doc: dict](abc.ABC):
    """
    Creates OpenSearch Bulk API create requests.
    """

    def __init__(self, docs: Iterable[RawDoc]):

        self.docs = docs

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

        yield i, current_request_body


class REST:
    def __init__(self, url: str, session: requests.Session):
        self.session = session
        self.url = url

    def bulk(self, body: str):
        """
        Send bulk indexing request.
        """

        final_url = f"{self.url}/_bulk"
        response = self.session.post(
            final_url, data=body, headers={"Content-Type": "application/x-ndjson"}
        )
        return response

    def delete_index(self, index: str):

        final_url = f"{self.url}/{index}"
        return self.session.delete(final_url)


def create_session(*, user_pass: tuple[str, str], verify_cert: bool):
    """
    Create a session suitable for connecting with a local OpenSearch REST API.
    """

    session = requests.Session()
    session.auth = user_pass
    session.verify = verify_cert
    return session


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
