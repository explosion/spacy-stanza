<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy + Stanza (formerly StanfordNLP)

This package wraps the [Stanza](https://github.com/stanfordnlp/stanza)
(formerly StanfordNLP) library, so you can use Stanford's models in a
[spaCy](https://spacy.io) pipeline. The Stanford models achieved top accuracy
in the CoNLL 2017 and 2018 shared task, which involves tokenization,
part-of-speech tagging, morphological analysis, lemmatization and labeled
dependency parsing in 68 languages. As of v1.0, Stanza also supports named
entity recognition for selected languages.

> ⚠️ Previous version of this package were available as
> [`spacy-stanfordnlp`](https://pypi.python.org/pypi/spacy-stanfordnlp).

[![Azure Pipelines](https://img.shields.io/azure-devops/build/explosion-ai/public/17/master.svg?logo=azure-pipelines&style=flat-square)](https://dev.azure.com/explosion-ai/public/_build?definitionId=17)
[![PyPi](https://img.shields.io/pypi/v/spacy-stanza.svg?style=flat-square)](https://pypi.python.org/pypi/spacy-stanza)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-stanza/all.svg?style=flat-square)](https://github.com/explosion/spacy-stanza)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

Using this wrapper, you'll be able to use the following annotations, computed by
your pretrained `stanza` model:

- Statistical tokenization (reflected in the `Doc` and its tokens)
- Lemmatization (`token.lemma` and `token.lemma_`)
- Part-of-speech tagging (`token.tag`, `token.tag_`, `token.pos`, `token.pos_`)
- Morphological analysis (`token.morph`)
- Dependency parsing (`token.dep`, `token.dep_`, `token.head`)
- Named entity recognition (`doc.ents`, `token.ent_type`, `token.ent_type_`, `token.ent_iob`, `token.ent_iob_`)
- Sentence segmentation (`doc.sents`)

## ️️️⌛️ Installation

As of v1.0.0 `spacy-stanza` is only compatible with **spaCy v3.x**. To install
the most recent version:

```bash
pip install spacy-stanza
```

For spaCy v2, install v0.2.x and refer to the [v0.2.x usage
documentation](https://github.com/explosion/spacy-stanza/tree/v0.2.x#-usage--examples):

```bash
pip install "spacy-stanza<0.3.0"
```

Make sure to also
[download](https://stanfordnlp.github.io/stanza/download_models.html) one of
the [pre-trained Stanza
models](https://stanfordnlp.github.io/stanza/models.html).

## 📖 Usage & Examples

> ⚠️ **Important note:** This package has been refactored to take advantage of
> [spaCy v3.0](https://spacy.io). Previous versions that were built for [spaCy
> v2.x](https://v2.spacy.io) worked considerably differently. Please see
> previous tagged versions of this README for documentation on prior versions.

Use `spacy_stanza.load_pipeline()` to create an `nlp` object that you can use to
process a text with a Stanza pipeline and create a spaCy [`Doc`
object](https://spacy.io/api/doc). By default, both the spaCy pipeline and the
Stanza pipeline will be initialized with the same `lang`, e.g. "en":

```python
import stanza
import spacy_stanza

# Download the stanza model if necessary
stanza.download("en")

# Initialize the pipeline
nlp = spacy_stanza.load_pipeline("en")

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc.ents)
```

If language data for the given language is available in spaCy, the respective
language class can be used as the base for the `nlp` object – for example,
`English()`. This lets you use spaCy's lexical attributes like `is_stop` or
`like_num`. The `nlp` object follows the same API as any other spaCy `Language`
class – so you can visualize the `Doc` objects with displaCy, add custom
components to the pipeline, use the rule-based matcher and do pretty much
anything else you'd normally do in spaCy.

```python
# Access spaCy's lexical attributes
print([token.is_stop for token in doc])
print([token.like_num for token in doc])

# Visualize dependencies
from spacy import displacy
displacy.serve(doc)  # or displacy.render if you're in a Jupyter notebook

# Process texts with nlp.pipe
for doc in nlp.pipe(["Lots of texts", "Even more texts", "..."]):
    print(doc.text)

# Combine with your own custom pipeline components
from spacy import Language
@Language.component("custom_component")
def custom_component(doc):
    # Do something to the doc here
    print(f"Custom component called: {doc.text}")
    return doc

nlp.add_pipe("custom_component")
doc = nlp("Some text")

# Serialize attributes to a numpy array
np_array = doc.to_array(['ORTH', 'LEMMA', 'POS'])
```

### Stanza Pipeline options

Additional options for the Stanza
[`Pipeline`](https://stanfordnlp.github.io/stanza/pipeline.html#pipeline) can be
provided as keyword arguments following the `Pipeline` API:

- Provide the Stanza language as `lang`. For Stanza languages without spaCy
  support, use "xx" for the spaCy language setting:

  ```python
  # Initialize a pipeline for Coptic
  nlp = spacy_stanza.load_pipeline("xx", lang="cop")
  ```

- Provide Stanza pipeline settings following the `Pipeline` API:

  ```python
  # Initialize a German pipeline with the `hdt` package
  nlp = spacy_stanza.load_pipeline("de", package="hdt")
  ```

- Tokenize with spaCy rather than the statistical tokenizer (only for English):

  ```python
  nlp = spacy_stanza.load_pipeline("en", processors= {"tokenize": "spacy"})
  ```

- Provide any additional processor settings as additional keyword arguments:

  ```python
  # Provide pretokenized texts (whitespace tokenization)
  nlp = spacy_stanza.load_pipeline("de", tokenize_pretokenized=True)
  ```

The spaCy config specifies all `Pipeline` options in the `[nlp.tokenizer]`
block. For example, the config for the last example above, a German pipeline
with pretokenized texts:

```ini
[nlp.tokenizer]
@tokenizers = "spacy_stanza.PipelineAsTokenizer.v1"
lang = "de"
dir = null
package = "default"
logging_level = null
verbose = null
use_gpu = true

[nlp.tokenizer.kwargs]
tokenize_pretokenized = true

[nlp.tokenizer.processors]
```

### Serialization

The full Stanza pipeline configuration is stored in the spaCy pipeline
[config](https://spacy.io/usage/training#config), so you can save and load the
pipeline just like any other `nlp` pipeline:

```python
# Save to a local directory
nlp.to_disk("./stanza-spacy-model")

# Reload the pipeline
nlp = spacy.load("./stanza-spacy-model")
```

Note that this **does not save any Stanza model data by default**. The Stanza
models are very large, so for now, this package expects you to download the
models separately with `stanza.download()` and have them available either in
the default model directory or in the path specified under
`[nlp.tokenizer.dir]` in the config.

### Adding additional spaCy pipeline components

By default, the spaCy pipeline in the `nlp` object returned by
`spacy_stanza.load_pipeline()` will be empty, because all `stanza` attributes
are computed and set within the custom tokenizer,
[`StanzaTokenizer`](spacy_stanza/tokenizer.py). But since it's a regular `nlp`
object, you can add your own components to the pipeline. For example, you could
add [your own custom text classification
component](https://spacy.io/usage/training) with `nlp.add_pipe("textcat",
source=source_nlp)`, or augment the named entities with your own rule-based
patterns using the [`EntityRuler`
component](https://spacy.io/usage/rule-based-matching#entityruler).
