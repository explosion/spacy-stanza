from typing import Optional, Union
from spacy import blank, Language

from . import tokenizer


def load_pipeline(
    name: str,
    *,
    lang: str = "",
    dir: Optional[str] = None,
    package: str = "default",
    processors: Union[dict, str] = {},
    logging_level: Optional[Union[int, str]] = None,
    verbose: Optional[bool] = None,
    use_gpu: bool = True,
    **kwargs,
) -> Language:
    """Create a blank nlp object for a given language code with a stanza
    pipeline as part of the tokenizer. To use the default stanza pipeline with
    the same language code, leave the tokenizer config empty. Otherwise, pass
    in the stanza pipeline settings in config["nlp"]["tokenizer"].

    name (str): The language code, e.g. "en".
    lang: str = "",
    dir: Optional[str] = None,
    package: str = "default",
    processors: Union[dict, str] = {},
    logging_level: Optional[Union[int, str]] = None,
    verbose: Optional[bool] = None,
    use_gpu: bool = True,
    **kwargs: Options for the individual stanza processors.
    RETURNS (Language): The nlp object.
    """
    # Create an empty config skeleton
    config = {"nlp": {"tokenizer": {"kwargs": {}}}}
    if lang == "":
        lang = name
    # Set the stanza tokenizer
    config["nlp"]["tokenizer"]["@tokenizers"] = "spacy_stanza.PipelineAsTokenizer.v1"
    # Set the stanza options
    config["nlp"]["tokenizer"]["lang"] = lang
    config["nlp"]["tokenizer"]["dir"] = dir
    config["nlp"]["tokenizer"]["package"] = package
    config["nlp"]["tokenizer"]["processors"] = processors
    config["nlp"]["tokenizer"]["logging_level"] = logging_level
    config["nlp"]["tokenizer"]["verbose"] = verbose
    config["nlp"]["tokenizer"]["use_gpu"] = use_gpu
    config["nlp"]["tokenizer"]["kwargs"].update(kwargs)
    return blank(name, config=config)
