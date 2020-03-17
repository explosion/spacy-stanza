<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy + Stanza (formerly StanfordNLP)

This package wraps the [Stanza](https://github.com/stanfordnlp/stanza)
(formerly StanfordNLP) library, so you can use Stanford's models as a
[spaCy](https://spacy.io) pipeline. The Stanford models achieved top accuracy in
the CoNLL 2017 and 2018 shared task, which involves tokenization,
part-of-speech tagging, morphological analysis, lemmatization and labelled
dependency parsing in 58 languages. As of v1.0, Stanza also supports named
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
- Dependency parsing (`token.dep`, `token.dep_`, `token.head`)
- Named entity recognition (`doc.ents`, `token.ent_type`, `token.ent_type_`, `token.ent_iob`, `token.ent_iob_`)
- Sentence segmentation (`doc.sents`)

## ️️️⌛️ Installation

```bash
pip install spacy-stanza
```

Make sure to also install one of the
[pre-trained Stanza models](https://stanfordnlp.github.io/stanza/models.html).

## 📖 Usage & Examples

The `StanzaLanguage` class can be initialized with a loaded Stanza
pipeline and returns a spaCy [`Language` object](https://spacy.io/api/language),
i.e. the `nlp` object you can use to process text and create a
[`Doc` object](https://spacy.io/api/doc).

```python
import stanza
from spacy_stanza import StanzaLanguage

snlp = stanza.Pipeline(lang="en")
nlp = StanzaLanguage(snlp)

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.ent_type_)
print(doc.ents)
```

If language data for the given language is available in spaCy, the respective
language class will be used as the base for the `nlp` object – for example,
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

# Efficient processing with nlp.pipe
for doc in nlp.pipe(["Lots of texts", "Even more texts", "..."]):
    print(doc.text)

# Combine with your own custom pipeline components
def custom_component(doc):
    # Do something to the doc here
    return doc

nlp.add_pipe(custom_component)

# Serialize it to a numpy array
np_array = doc.to_array(['ORTH', 'LEMMA', 'POS'])
```

### Experimental: Mixing and matching pipeline components

By default, the `nlp` object's pipeline will be empty, because all attributes
are computed once and set in the custom [`Tokenizer`](spacy_stanza/language.py).
But since it's a regular `nlp` object, you can add your own components to the
pipeline. For example, you could add and train
[your own custom text classification component](https://spacy.io/usage/training#textcat)
and use `nlp.add_pipe` to add it to the pipeline, or augment the named
entities with your own rule-based patterns using the
[`EntityRuler` component](https://spacy.io/usage/rule-based-matching#entityruler).

### Advanced: serialization and entry points

The spaCy `nlp` object created by `StanzaLanguage` exposes its language as
`stanza_xx`.

```python
from spacy.util import get_lang_class
lang_cls = get_lang_class("stanza_en")
```

Normally, the above would fail because spaCy doesn't include a language class
`stanza_en`. But because this package exposes a `spacy_languages` entry
point in its [`setup.py`](setup.py) that points to `StanzaLanguage`, spaCy
knows how to initialize it.

This means that saving to and loading from disk works:

```python
snlp = stanza.Pipeline(lang="en")
nlp = StanzaLanguage(snlp)
nlp.to_disk("./stanza-spacy-model")
```

Additional arguments on `spacy.load` are automatically passed down to the
language class and pipeline components. So when loading the saved model, you can
pass in the `snlp` argument:

```python
snlp = stanza.Pipeline(lang="en")
nlp = spacy.load("./stanza-spacy-model", snlp=snlp)
```

Note that this **will not save any model data by default**. The Stanza
models are very large, so for now, this package expects that you load them
separately.
