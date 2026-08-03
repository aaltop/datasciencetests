import abc
import contextlib
import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import requests

from src import filesystem
from src.env import Env


class BaseREST[ResponseType](abc.ABC):
    @abc.abstractmethod
    def bulk(self, body: str) -> ResponseType:
        """
        Send bulk indexing request.
        """

    @abc.abstractmethod
    def delete_index(self, index: str) -> ResponseType: ...


class RESTState:
    def __init__(self, url: str, session: requests.Session):
        self.session = session
        self.base_url = url


class IndexREST(RESTState):
    def index_url(self, index_name: str):

        return f"{self.base_url}/{index_name}"

    def search_index(
        self,
        index_name: str,
        *,
        semantic_query: str | None = None,
        match_query: str | None = None,
    ):

        if (semantic_query is None) and (match_query is None):
            raise ValueError(
                "One of 'semantic_query' and 'match_query' should be not None."
            )

        final_url = self.index_url(index_name) + "/_search"
        data = default_index_search_body(semantic_query)
        data["_source"] = {"excludes": ["text_embedding"]}
        data = json.dumps(data)

        response = self.session.post(
            final_url, data, headers={"Content-Type": "application/json"}
        )
        return response


class ModelsREST(RESTState):
    def models_url(self, suffix: str):

        return f"{self.base_url}/_plugins/_ml/models/{suffix}"

    def get_models(self):

        data = json.dumps(dict(query=dict(match_all={})))
        response = self.session.post(
            url=self.models_url("_search"),
            data=data,
            headers={"Content-Type": "application/json"},
        )
        return response

    def get_model(self, model_id: str):

        response = self.session.get(url=self.models_url(model_id))
        return response

    def deploy_model(self, model_id: str):

        return self.session.post(url=self.models_url(model_id + "/_deploy"))

    def undeploy_model(self, model_id: str):

        return self.session.post(url=self.models_url(model_id + "/_undeploy"))


class ConnectorsREST(RESTState):
    def connectors_url(self, suffix: str):

        return f"{self.base_url}/_plugins/_ml/connectors/{suffix}"

    def modify_connector(self, connector_id: str, data: dict):

        response = self.session.put(
            url=self.connectors_url(connector_id),
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )
        return response


class REST(BaseREST[requests.Response], ModelsREST, ConnectorsREST, IndexREST):
    def bulk(self, body: str):

        final_url = f"{self.base_url}/_bulk"
        response = self.session.post(
            final_url, data=body, headers={"Content-Type": "application/x-ndjson"}
        )
        return response

    def delete_index(self, index: str):

        final_url = f"{self.base_url}/{index}"
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


class IndexSearchBody[ExcludeLiterals: Literal](TypedDict):
    _source: NotRequired[Source[ExcludeLiterals]]

    query: Query

    class Source[T](TypedDict):
        excludes: list[T]

    class Query(TypedDict):
        neural: IndexSearchBody.Neural

    class Neural(TypedDict):
        text_embedding: IndexSearchBody.TextEmbedding

    class TextEmbedding(TypedDict):
        query_text: str
        k: int


def default_index_search_body(semantic_query: str):

    return IndexSearchBody(
        query={"neural": {"text_embedding": {"query_text": semantic_query, "k": 5}}},
    )
