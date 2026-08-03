import abc
import json
from collections.abc import Generator
from typing import NotRequired, TypedDict, Unpack


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
