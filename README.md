<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spaCy + StanfordNLP

This package...

## Usage

```python
import stanfordnlp
from spacy_stanfordnlp import StanfordNLPLanguage

snlp = stanfordnlp.Pipeline(lang="en")
nlp = StanfordNLPLanguage(snlp)

doc = nlp("Barack Obama was born in Hawaii. He was elected president in 2008.")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_, token.is_sent_start)

# Access spaCy's lexical attributes and syntax iterators
print([token.is_stop for token in doc])
print(doc.noun_chunks)

# Visualize dependencies
from spacy import displacy
displacy.serve(doc)  # or displacy.render if you're in a Jupyter notebook
```

## 🎛 API

The `StanfordNLPLanguage` is initialized with the `stanfordnlp.Pipeline` object
and otherwise follows the same API as `spacy.language.Language`.
