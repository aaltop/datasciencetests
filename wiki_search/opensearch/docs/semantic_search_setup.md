Following largely [the tutorial](https://docs.opensearch.org/latest/tutorials/vector-search/neural-search-tutorial/).

# Semantic Embedding Model

Adding a semantic embedding model to OpenSearch. The intention is to use the
[paraphrase-multilingual-MiniLM-L12-v2 model](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
but with some changes.
See also [other supported semantic models](https://docs.opensearch.org/latest/ml-commons-plugin/pretrained-models/#sentence-transformers).

```
POST /_plugins/_ml/models/_register?deploy=true
{
  "name": "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "version": "1.0.2",
  "model_format": "TORCH_SCRIPT"
}
```

This should return a `task_id`, which can be used to query for the task
which is registering the model:

```
GET /_plugins/_ml/tasks/<model_id>
```

Once the model has been created, it can be fetched using the `model_id`, which
can be found using the dashboard, under 'Machine Learning' in 'OpenSearch Plugins',
and is also in the response of the task GET (see above).

```
GET /_plugins/_ml/models/<model_id>
```

Deploy (might already be deployed due to the URL param in the first step):

```
POST /_plugins/_ml/models/<model_id>/_deploy
```

It [can be tested:](https://docs.opensearch.org/latest/ml-commons-plugin/pretrained-models/#step-4-optional-test-the-model)

```
POST /_plugins/_ml/_predict/text_embedding/<model_id>
{
  "text_docs":[ "today is sunny"]
}
```

This is particularly for models whose 'algorithm' field is 'TEXT_EMBEDDING'.

# Ingest Pipeline

```
PUT /_ingest/pipeline/<pipeline_name>
{
  "description": "An ingest pipeline",
  "processors": [
    {
      "text_embedding": {
        "model_id": "<model_id>",
        "field_map": {
          "<field_to_map>": "<field_to_store_mapping_in>"
        }
      }
    }
  ]
}
```

# Semantic Search Index

Creating an index for use with semantic search. The dimension of the embedding
field depends on the model used: here, the intention is to use the
[paraphrase--multilingual-MiniLM-L12-v2 model](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2),
which uses a 384-dimensional vector space.

```
PUT /semantic_wiki_pages
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text"
      },
      "text": {
        "type": "text"
      },
      "text_embedding": {
        "type": "knn_vector",
        "dimension": 384,
        "space_type": "l2"
      }
    }
  }
}
```

Setting a default (ingest) pipeline for the index (note that this is
different from a search pipeline):

```
PUT /<index_name>/_settings
{
  "index.default_pipeline" : "<pipeline_name>"
}
```

Could also be done when creating the index, or in the 'Advanced settings' section
of the index once it is created.

# Ingesting data

Once the previous steps have been completed, it is possible to ingest
data normally into the index and the data will be processed by the
semantic embedding model, allowing for semantic search. Alternatively,
if an index with data is already available, it is possible to [reindex](https://docs.opensearch.org/latest/im-plugin/reindex-data/)
data, ingesting data from the old index to the new, semantic embedding -ready
index. Test with a smaller subset of the source index by only reindexing
documents matching a search query:

```
POST _reindex
{
  "source":{
    "index":"<source_index>",
    "query": {
      "match": {
          "<source_index_field>": "<search_term>"
        }
    }
  },
  "dest":{
    "index":"<destination_index>"
  }
}
```

Search using semantic search:

```
GET /<semantic_index_name>/_search
{
  "_source": {
    "excludes": [
      <excluded_fields...>
    ]
  },
  "query": {
    "neural": {
      "<field_to_query_on (knn_vector field)>": {
        "query_text": "<semantic_query>",
        "model_id": "<semantic_embedding_model_id>",
        "k": <integer>
      }
    }
  }
}
```

For a quick way to delete the reindexed data, assuming not too much data
indexed:

```
POST <semantic_index_name>/_delete_by_query
{
  "query": {
    "match_all": {

    }
  }
}
```

To reindex the whole source index:

```
POST _reindex
{
   "source":{
      "index":"<source_index>"
   },
   "dest":{
      "index":"<destination_index>"
   }
}
```

# Links

Tutorial https://docs.opensearch.org/latest/tutorials/vector-search/neural-search-tutorial/

Supported semantic models https://docs.opensearch.org/latest/ml-commons-plugin/pretrained-models/#sentence-transformers
