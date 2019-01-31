from spacy.symbols import POS, TAG, DEP, LEMMA, HEAD
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.util import get_lang_class
import numpy


class StanfordNLPLanguage(Language):
    def __init__(self, snlp, max_length=10 ** 6):
        """Initialize the Language class.

        snlp (stanfordnlp.Pipeline): The loaded StanfordNLP pipeline.
        max_length (int): Maximum document length, for consistency.
        RETURNS (spacy.language.Language): The nlp object.
        """
        self.snlp = snlp
        self.lang = snlp.config["lang"]
        self.Defaults = get_defaults(self.lang)
        self.vocab = self.Defaults.create_vocab()
        self._meta = dict(snlp.config)
        self._path = None
        self.pipeline = []
        self.max_length = max_length
        self._optimizer = None

    def make_doc(self, text):
        snlp_doc = self.snlp(text)
        return get_doc(snlp_doc, self.vocab)

    def tokenizer(self, text):
        return self.make_doc(text)


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
    except:
        return Language.Defaults


def get_tokens_with_heads(snlp_doc):
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
                tokens.append(word)
                # TODO: fix heads – they somehow don't match what spaCy expects?
                heads.append(word.governor + offset if word.governor is not None else 0)
        offset += len(sentence.tokens)
    return tokens, heads


def get_doc(snlp_doc, vocab):
    """Convert a StanfordNLP Doc to a spaCy Doc.

    snlp_doc (stanfordnlp.Document): The processed StanfordNLP doc.
    vocab (spacy.vocab.Vocab): The vocabulary to use.
    RETURNS (spacy.tokens.Doc): The spaCy Doc object.
    """
    text = snlp_doc.text
    tokens, heads = get_tokens_with_heads(snlp_doc)
    if not len(tokens):
        raise ValueError("No tokens available")
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
        pos.append(vocab.strings.add(token.upos or ""))
        tags.append(vocab.strings.add(token.xpos or ""))
        deps.append(vocab.strings.add(token.dependency_relation or ""))
        lemmas.append(vocab.strings.add(token.lemma or ""))
        offset += len(token.text)
        span = text[offset:]
        if i == len(tokens) - 1:
            spaces.append(False)
        else:
            next_token = tokens[i + 1]
            spaces.append(not span.startswith(next_token.text))
    attrs = [POS, TAG, DEP, HEAD, LEMMA]
    array = numpy.array(list(zip(pos, tags, deps, heads, lemmas)), dtype="uint64")
    doc = Doc(vocab, words=words, spaces=spaces).from_array(attrs, array)
    if any(pos) and any(tags):
        doc.is_tagged = True
    if any(deps):
        doc.is_parsed = True
    return doc
