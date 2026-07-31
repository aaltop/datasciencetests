# Wiki Search

Setup for testing OpenSearch with [data from Wikipedia](https://dumps.wikimedia.org/),
specifically using the XML dumps such as [the current ones](https://dumps.wikimedia.org/other/mediawiki_content_current/).

## Scripts

- [Parse pages from the XML files](./parse_pages_from_xml.py)

- [Create semantic embeddings for the documents](./semantic_embedding.py)

- [Send documents to OpenSearch](./send_to_opensearch.py)

- [Serve embedding model](./serve_text_embedding_model.py)

## OpenSearch Configuration

Run `docker compose up` on [the compose file](./opensearch/docker/docker-compose.yml).
See [the instructions](./opensearch/docs/) for configuring things
in OpenSearch.

## Certificates

OpenSearch by default creates self-signed certificates for development
testing. Getting the certificate from the cluster and placing it under
`certs/root_ca.pem` should allow for TLS verification to work. Note that
this is just for testing/development setups.

## Environment variables

For starting OpenSearch on Docker, an .env file should be created in
[the opensearch docker directory](./opensearch/docker/), containing
a password under `OPENSEARCH_INITIAL_ADMIN_PASSWORD`. This password
should also be set in the env.toml file in the project root; see
[the example file for more](/example_env.toml).