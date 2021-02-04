from typing import Any, Dict, Union
from copy import deepcopy
from thinc.api import Config
from spacy import blank as spacy_blank, util, Language, Vocab

from .tokenizer import DEFAULT_TOKENIZER_CONFIG


def blank(
    name: str,
    *,
    vocab: Union[Vocab, bool] = True,
    config: Union[Dict[str, Any], Config] = util.SimpleFrozenDict(),
    meta: Dict[str, Any] = util.SimpleFrozenDict()
) -> Language:
    """Create a blank nlp object for a given language code with a stanza
    pipeline as part of the tokenizer. To use the default stanza pipeline with
    the same language code, leave the tokenizer config empty. Otherwise, pass
    in the stanza pipeline settings in config["nlp"]["tokenizer"].

    name (str): The language code, e.g. "en".
    vocab (Vocab): A Vocab object. If True, a vocab is created.
    config (Dict[str, Any] / Config): Optional config overrides.
    meta (Dict[str, Any]): Overrides for nlp.meta.
    RETURNS (Language): The nlp object.
    """
    # We should accept both dot notation and nested dict here for consistency
    config = util.dot_to_dict(config)
    # Copy any config settings into a new editable config, adding an
    # nlp.tokenizer block if necessary
    stanza_config = {}
    stanza_config.update(deepcopy(config))
    if "nlp" not in stanza_config:
        stanza_config["nlp"] = {}
    if "tokenizer" not in stanza_config["nlp"]:
        stanza_config["nlp"]["tokenizer"] = {}
    # Replace the nlp.tokenizer block with the default stanza config
    stanza_config["nlp"]["tokenizer"] = deepcopy(DEFAULT_TOKENIZER_CONFIG)
    # Use the same language code for stanza by default
    stanza_config["nlp"]["tokenizer"]["lang"] = name
    # Update the default stanza config with any user-provided settings
    stanza_config["nlp"]["tokenizer"].update(
        deepcopy(config.get("nlp", {}).get("tokenizer", {}))
    )
    return spacy_blank(name, vocab=vocab, config=stanza_config, meta=meta)
