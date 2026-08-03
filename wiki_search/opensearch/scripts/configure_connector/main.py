"""
This script configures an OpenSearch connector to have a pre-processing
function that allows both normal predictions (using the _predict endpoint)
and neural search predictions. The former sends the query as params.text_docs,
while the latter sends it as params.input, and the Painless script combines
these two possibilities, sending out the correct request. Note that the
'correct' request depends on the model that the connector connects to:
here, that model's server expects a POST body of { text_docs: list<str> },
which the connector creates based on the content of the request_body
field, where it sets the templated content based on the output of the
pre_process_function.

As a side note, it doesn't seem that the _predict endpoint actually uses
the pre-processing function anyway, though this is not entirely clear. It
does seem to work all the same.

Note that the model_id and connector_id should be set in the directory
where this script is, in an env.toml file. The model_id and (un)deploy
is, however, only for if a model has been connected to the connector,
and you wish to deploy the model automatically after the connector has
been configured by running this script. In other words, if you only have
the connector setup in OpenSearch, it is fine to comment out and ignore
any model-related code.
"""

import sys
import tomllib
from pathlib import Path

sys.path.append(str(Path().absolute()))


from src.opensearch import rest

painless_script = r"""
def input = params.text_docs != null
    ? params.text_docs
    : params.input;
for (int i; i < input.size(); ++i) {
    input[i] = "\"" + input[i] + "\"";
}
return "{ \"parameters\": { \"text_docs\": [" + String.join(",", input) + "] } }";"""


def some():

    local_root = Path(__file__).parent
    env = tomllib.loads((local_root / "env.toml").read_text())

    connector_put = {
        "actions": [
            {
                "action_type": "predict",
                "method": "POST",
                "url": "${parameters.endpoint}/predict",
                "request_body": '{ "text_docs": ${parameters.text_docs} }',
                "pre_process_function": painless_script,
            }
        ]
    }

    model_id = env["model_id"]
    connector_id = env["connector_id"]
    with rest.default_rest() as api:
        api.undeploy_model(model_id)
        response = api.modify_connector(connector_id, connector_put)
        api.deploy_model(model_id)
        print(response.text)


if __name__ == "__main__":
    some()
