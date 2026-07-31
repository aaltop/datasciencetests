Following the following guide: https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/index/

The intention is to serve a model locally to OpenSearch. The model is
exposed through a locally run server endpoint.

# Setting trusted connector endpoints

Do a catch-all localhost trusted connector endpoints, and set private IPs
to enabled to allow localhost:
```
PUT /_cluster/settings
{
    "persistent": {
        "plugins.ml_commons.trusted_connector_endpoints_regex": [
          "^http://host.docker.internal.*"
        ],
        "plugins.ml_commons.connector.private_ip_enabled": true
    }
}
```

host.docker.internal refers to the host machine on which docker is running.

https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/index/#adding-trusted-endpoints
https://opensearch.org/blog/connect-opensearch-to-private-ml-endpoints/

# Create a model group

Model groups allow grouping models.

```
POST /_plugins/_ml/model_groups/_register
{
  "name": "local_models",
  "description": "A model group for local models"
}
```

For finding the model group ID later, look through the groups:
```
POST /_plugins/_ml/model_groups/_search
{
  "query": {
    "match_all": {}
  }
}
```

https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/index/#step-1-register-a-model-group 
https://docs.opensearch.org/latest/ml-commons-plugin/api/model-group-apis/index/

# Create a connector

The "credential" field is required, but its values can be presumably left
unused if not needed. Here, the Authorization header is left for
illustrative purposes, but can be removed if the server used doesn't
use authorisation (which the server here (see below) doesn't).
```
POST /_plugins/_ml/connectors/_create
{
    "name": "local_minilm_text_embedding",
    "description": "Locally served MiniLM model for text embedding creation.",
    "version": 1,
    "protocol": "http",
    "parameters": {
        "endpoint": "http://host.docker.internal:8000",
    },
    "credential": {
        "optional_credential": "pass"
    },
    "actions": [
        {
            "action_type": "predict",
            "method": "POST",
            "url": "${parameters.endpoint}/predict",
            "headers": {
                "Authorization": "Bearer ${credential.optional_credential}"
            },
            "request_body": "{ \"text_docs\": ${parameters.text_docs} }"
        }
    ]
}
```

Note that the `models/<model_id>/_predict` query uses `parameters.text_docs`,
while using the model directly in a query ('neural search')
requires `parameters.input`.
It may ideal to set `pre_process_function` using [the script](../scripts/configure_connector/main.py),
which should allow neural search to also work. The server for the model
itself can be served by running `uv run fastapi dev serve_text_embedding_model.py`.
Note that the `parameters.endpoint` in the _create call naturally
depends on the settings of this model server. 

https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/index/#step-2-create-a-connector

https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/blueprints/

https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/blueprints/#custom-pre--and-post-processing-functions

See https://commons.apache.org/proper/commons-text/apidocs/org/apache/commons/text/StringSubstitutor.html
For the template syntax used in the request.


# Model setup

Register a model, noting the returned model ID (or use the '_search' path
for models with 'match_all' query):
```
POST /_plugins/_ml/models/_register
{
    "name": "minilm-text-embedding",
    "function_name": "remote",
    "model_group_id": "<model_group_id>",
    "description": "Text embedding model.",
    "connector_id": "<connector_id>"
}
```

See the model, and deploy it once it's registered:
```
GET /_plugins/_ml/models/<model_id>

POST /_plugins/_ml/models/<model_id>/_deploy
```

The model won't show up in the dashboard until it's deployed. Search
for the model in the REST API:
```
POST /_plugins/_ml/models/_search
{
  "query": {
    "match": {
      "name": "text-embedding"
    }
  }
}
```

A few useful commands:
```
GET /_plugins/_ml/models/<model_id>

POST /_plugins/_ml/models/<model_id>/_deploy

POST /_plugins/_ml/models/<model_id>/_undeploy

POST /_plugins/_ml/models/<model_id>/_predict
{
  "parameters: {
    "text_docs": [
      "This a query",
      "predict this, please!"
    ]
  }
}

POST /semantic_wiki_pages/_search
{
  "_source": {
    "excludes": [
      "text_embedding",
      "text"
    ]
  },
  "query": {
    "neural": {
      "text_embedding": {
        "query_text": "<search term>",
        "model_id": "<model_id>",
        "k": 5
      }
    }
  }
}
```

Note that the last call (the POST to _search) requires the index `semantic_wiki_pages` whose
documents have a text embedding (vector embedding) field `text_embedding`,
and here also `text`, as the `excludes` implies.
The text embeddings should be calculated using the same model that is
being set up here so that the embeddings of the documents and of the
one created from the query text match.

See [the server](../../serve_text_embedding_model.py) for serving the
endpoint.