from typing import Any, Dict, Union
from copy import deepcopy
from thinc.api import Config
from spacy import blank as spacy_blank, util, Language, Vocab

from . import tokenizer


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
    if "nlp" not in config:
        config["nlp"] = {}
    if "tokenizer" not in config["nlp"]:
        config["nlp"]["tokenizer"] = {}
    # Set the stanza tokenizer
    config["nlp"]["tokenizer"]["@tokenizers"] = "spacy_stanza.PipelineAsTokenizer.v1"
    # Set the stanza lang if not provided
    if "lang" not in config["nlp"]["tokenizer"]:
        config["nlp"]["tokenizer"]["lang"] = name
    return spacy_blank(name, vocab=vocab, config=config, meta=meta)
