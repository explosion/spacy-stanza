<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy + StanfordNLP

This package wraps the [StanfordNLP](https://github.com/stanfordnlp/stanfordnlp)
library, so you can use Stanford's models as a [spaCy](https://spacy.io)
pipeline. The Stanford models achieved top accuracy in the CoNLL 2017 and 2018
shared task, which involves tokenization, part-of-speech tagging, morphological
analysis, lemmatization and labelled dependency parsing in 58 languages.

[![PyPi](https://img.shields.io/pypi/v/spacy-stanfordnlp.svg?style=flat-square)](https://pypi.python.org/pypi/spacy-stanfordnlp)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-stanfordnlp/all.svg?style=flat-square)](https://github.com/explosion/spacy-stanfordnlp)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

Using this wrapper, you'll be able to use the following annotations, computed by
your pretrained `stanfordnlp` model:

- Statistical tokenization (reflected in the `Doc` and its tokens)
- Lemmatization (`token.lemma` and `token.lemma_`)
- Part-of-speech tagging (`token.tag`, `token.tag_`, `token.pos`, `token.pos_`)
- Dependency parsing (`token.dep`, `token.dep_`, `token.head`)
- Sentence segmentation (`doc.sents`)

## ️️️⌛️ Installation

```bash
pip install spacy-stanfordnlp
```

Make sure to also install one of the
[pre-trained StanfordNLP models](https://stanfordnlp.github.io/stanfordnlp/installation_download.html). It's recommended to run StanfordNLP on Python 3.6.8+ or Python 3.7.2+.

## 📖 Usage & Examples

The `StanfordNLPLanguage` class can be initialized with a loaded StanfordNLP
pipeline and returns a spaCy [`Language` object](https://spacy.io/api/language),
i.e. the `nlp` object you can use to process text and create a
[`Doc` object](https://spacy.io/api/doc).

```python
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage

snlp = stanfordnlp.Pipeline(lang="en")
nlp = StanfordNLPLanguage(snlp)

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)
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
are computed once and set in the custom
[`Tokenizer`](spacy_stanfordnlp/language.py). But since it's a regular `nlp`
object, you can add your own components to the pipeline.

For example, the entity recognizer from one of spaCy's pre-trained models:

```python
import spacy
import spacy_stanfordnlp
import stanfordnlp

snlp = stanfordnlp.Pipeline(lang="en", models_dir="./models")
nlp = StanfordNLPLanguage(snlp)

# Load spaCy's pre-trained en_core_web_sm model, get the entity recognizer and
# add it to the StanfordNLP model's pipeline
spacy_model = spacy.load("en_core_web_sm")
ner = spacy_model.get_pipe("ner")
nlp.add_pipe(ner)

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
print([(ent.text, ent.label_) for ent in doc.ents])
# [('Barack Obama', 'PERSON'), ('Hawaii', 'GPE'), ('2008', 'DATE')]
```

You could also add and train
[your own custom text classification component](https://spacy.io/usage/training#textcat).

### Advanced: serialization and entry points

The spaCy `nlp` object created by `StanfordNLPLanguage` exposes its language as
`stanfordnlp_xx`.

```python
from spacy.util import get_lang_class
lang_cls = get_lang_class("stanfordnlp_en")
```

Normally, the above would fail because spaCy doesn't include a language class
`stanfordnlp_en`. But because this package exposes a `spacy_languages` entry
point in its [`setup.py`](setup.py) that points to `StanfordNLPLanguage`, spaCy
knows how to initialize it.

This means that saving to and loading from disk works:

```python
snlp = stanfordnlp.Pipeline(lang="en")
nlp = StanfordNLPLanguage(snlp)
nlp.to_disk("./stanfordnlp-spacy-model")
```

Additional arguments on `spacy.load` are automatically passed down to the
language class and pipeline components. So when loading the saved model, you can
pass in the `snlp` argument:

```python
snlp = stanfordnlp.Pipeline(lang="en")
nlp = spacy.load("./stanfordnlp-spacy-model", snlp=snlp)
```

Note that this **will not save any model data by default**. The StanfordNLP
models are very large, so for now, this package expects that you load them
separately.
