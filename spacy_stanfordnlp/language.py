# coding: utf8
from spacy.symbols import POS, TAG, DEP, LEMMA, HEAD
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.util import get_lang_class
import numpy


class StanfordNLPLanguage(Language):
    def __init__(self, snlp, meta=None, **kwargs):
        """Initialize the Language class.

        Instead of "en" etc. we call the language "stanfordnlp_en" to not
        cause conflicts with spaCy's built-in languages. Using entry points,
        this also allows serializing and deserializing the language class
        and "lang": "stanfordnlp_en" in the meta.json will automatically
        instantiate this class if this package is available.

        snlp (stanfordnlp.Pipeline): The loaded StanfordNLP pipeline.
        kwargs: Optional config parameters.
        RETURNS (spacy.language.Language): The nlp object.
        """
        lang = snlp.config["lang"]
        self.lang = "stanfordnlp_" + lang
        self.Defaults = get_defaults(lang)
        self.vocab = self.Defaults.create_vocab()
        self.tokenizer = Tokenizer(snlp, self.vocab)
        self.pipeline = []
        self.max_length = kwargs.get("max_length", 10 ** 6)
        self._meta = (
            {"lang": self.lang, "stanfordnlp": snlp.config}
            if meta is None
            else dict(meta)
        )
        self._path = None
        self._optimizer = None

    def make_doc(self, text):
        return self.tokenizer(text)


def get_defaults(lang):
    """Get the language-specific defaults, if available in spaCy. This allows
    using lexical attribute getters that depend in static language data, e.g.
    Token.like_num, Token.is_stop, Doc.noun_chunks etc.

    lang (unicode): The language code.
    RETURNS (Language.Defaults): The language defaults.
    """
    try:
        lang_cls = get_lang_class(lang)
        return lang_cls.Defaults
    except ImportError:
        return Language.Defaults


class Tokenizer(object):
    """Because we're only running the StanfordNLP pipeline once and don't split
    it up into spaCy pipeline components, we'll set all the attributes within
    a custom tokenizer. The tokenizer is currently expected to
    implement serialization methods so we're mocking them up here. When loading
    the serialized nlp object back in, you can pass the `snlp` to spacy.load:

    >>> nlp.to_disk('/path/to/model')
    >>> nlp = spacy.load('/path/to/model', snlp=snlp)
    """

    to_disk = lambda self, *args, **kwargs: None
    from_disk = lambda self, *args, **kwargs: None
    to_bytes = lambda self, *args, **kwargs: None
    from_bytes = lambda self, *args, **kwargs: None

    def __init__(self, snlp, vocab):
        """Initialize the tokenizer.

        snlp (stanfordnlp.Pipeline): The initialized StanfordNLP pipeline.
        vocab (spacy.vocab.Vocab): The vocabulary to use.
        RETURNS (Tokenizer): The custom tokenizer.
        """
        self.snlp = snlp
        self.vocab = vocab

    def __call__(self, text):
        """Convert a StanfordNLP Doc to a spaCy Doc.

        text (unicode): The text to process.
        RETURNS (spacy.tokens.Doc): The spaCy Doc object.
        """
        snlp_doc = self.snlp(text)
        text = snlp_doc.text
        tokens, heads = self.get_tokens_with_heads(snlp_doc)
        if not len(tokens):
            raise ValueError("No tokens available.")
        words = []
        spaces = []
        pos = []
        tags = []
        deps = []
        lemmas = []
        offset = 0
        for i, token in enumerate(tokens):
            span = text[offset:]
            if not len(span):
                break
            while not span.startswith(token.text):
                # If we encounter leading whitespace, skip one character ahead
                offset += 1
                span = text[offset:]
            words.append(token.text)
            # Make sure all strings are in the vocabulary
            pos.append(self.vocab.strings.add(token.upos or ""))
            tags.append(self.vocab.strings.add(token.xpos or ""))
            deps.append(self.vocab.strings.add(token.dependency_relation or ""))
            lemmas.append(self.vocab.strings.add(token.lemma or ""))
            offset += len(token.text)
            span = text[offset:]
            if i == len(tokens) - 1:
                spaces.append(False)
            else:
                next_token = tokens[i + 1]
                spaces.append(not span.startswith(next_token.text))
        attrs = [POS, TAG, DEP, HEAD, LEMMA]
        array = numpy.array(list(zip(pos, tags, deps, heads, lemmas)), dtype="uint64")
        doc = Doc(self.vocab, words=words, spaces=spaces).from_array(attrs, array)
        if any(pos) and any(tags):
            doc.is_tagged = True
        if any(deps):
            doc.is_parsed = True
        return doc

    def get_tokens_with_heads(self, snlp_doc):
        """Flatten the tokens in the StanfordNLP Doc and extract the token indices
        of the sentence start tokens to set is_sent_start.

        snlp_doc (stanfordnlp.Document): The processed StanfordNLP doc.
        RETURNS (list): The tokens (words).
        """
        tokens = []
        heads = []
        offset = 0
        for sentence in snlp_doc.sentences:
            for token in sentence.tokens:
                for word in token.words:
                    # Here, we're calculating the absolute token index in the doc,
                    # then the *relative* index of the head, -1 for zero-indexed
                    # and if the governor is 0 (root), we leave it at 0
                    if word.governor:
                        head = word.governor + offset - len(tokens) - 1
                    else:
                        head = 0
                    heads.append(head)
                    tokens.append(word)
            offset += sum(len(token.words) for token in sentence.tokens)
        return tokens, heads
