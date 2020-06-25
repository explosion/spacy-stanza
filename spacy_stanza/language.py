# coding: utf8
from spacy.symbols import POS, TAG, DEP, LEMMA, HEAD
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import get_lang_class

from stanza.models.common.vocab import UNK_ID
from stanza.models.common.pretrain import Pretrain
from stanza import Document

import numpy
import re
import warnings


class StanzaLanguage(Language):
    def __init__(self, snlp, meta=None, **kwargs):
        """Initialize the Language class.

        Instead of "en" etc. we call the language "stanza_en" to not
        cause conflicts with spaCy's built-in languages. Using entry points,
        this also allows serializing and deserializing the language class
        and "lang": "stanza_en" in the meta.json will automatically
        instantiate this class if this package is available.

        snlp (stanza.Pipeline): The loaded Stanza pipeline.
        kwargs: Optional config parameters.
        RETURNS (spacy.language.Language): The nlp object.
        """
        if hasattr(snlp, 'lang'):
            lang = snlp.lang
        else:
            # backward compatible with stanza v1.0.0
            lang = snlp.processors["tokenize"].config["lang"]
        self.snlp = snlp
        self.svecs = StanzaLanguage._find_embeddings(snlp)
        self.lang = "stanza_" + lang
        self.Defaults = get_defaults(lang)
        self.vocab = self.Defaults.create_vocab()
        self.tokenizer = Tokenizer(snlp, self.vocab)
        self.pipeline = []
        self.max_length = kwargs.get("max_length", 10 ** 6)
        self._meta = (
            {"lang": self.lang, "stanza": snlp.config} if meta is None else dict(meta)
        )
        self._path = None
        self._optimizer = None

    @staticmethod
    def _find_embeddings(snlp):
        """Find pretrained word embeddings in any of a SNLP's processors.

        RETURNS (Pretrain): Or None if no embeddings were found.
        """
        embs = None
        for proc in snlp.processors.values():
            if hasattr(proc, "pretrain") and isinstance(proc.pretrain, Pretrain):
                embs = proc.pretrain
                break
        return embs

    def make_doc(self, text):
        """Execute Stanza pipeline on text and extract attributes into Spacy Doc.
        If the Stanza pipeline contains a processor with pretrained word embeddings
        these will be mapped to token vectors.
        """
        doc = self.tokenizer(text)
        if self.svecs is not None:
            doc.user_token_hooks["vector"] = self.token_vector
            doc.user_token_hooks["has_vector"] = self.token_has_vector
        return doc

    def token_vector(self, token):
        """Get Stanza's pretrained word embedding for given token.

        token (Token): The token whose embedding will be returned
        RETURNS (np.ndarray[ndim=1, dtype='float32']): the embedding/vector.
            token.vector.size > 0 if Stanza pipeline contains a processor with
            embeddings, else token.vector.size == 0. A 0-vector (origin) will be returned
            when the token doesn't exist in snlp's pretrained embeddings."""
        unit_id = self.svecs.vocab.unit2id(token.text)
        return self.svecs.emb[unit_id]

    def token_has_vector(self, token):
        """Check if the token exists as a unit in snlp's pretrained embeddings."""
        return self.svecs.vocab.unit2id(token.text) != UNK_ID


def get_defaults(lang):
    """Get the language-specific defaults, if available in spaCy. This allows
    using lexical attribute getters that depend on static language data, e.g.
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
    """Because we're only running the Stanza pipeline once and don't split
    it up into spaCy pipeline components, we'll set all the attributes within
    a custom tokenizer. The tokenizer is currently expected to
    implement serialization methods so we're mocking them up here. When loading
    the serialized nlp object back in, you can pass `snlp` to spacy.load:

    >>> nlp.to_disk('/path/to/model')
    >>> nlp = spacy.load('/path/to/model', snlp=snlp)
    """

    to_disk = lambda self, *args, **kwargs: None
    from_disk = lambda self, *args, **kwargs: None
    to_bytes = lambda self, *args, **kwargs: None
    from_bytes = lambda self, *args, **kwargs: None
    _ws_pattern = re.compile(r"\s+")

    def __init__(self, snlp, vocab):
        """Initialize the tokenizer.

        snlp (stanza.Pipeline): The initialized Stanza pipeline.
        vocab (spacy.vocab.Vocab): The vocabulary to use.
        RETURNS (Tokenizer): The custom tokenizer.
        """
        self.snlp = snlp
        self.vocab = vocab

    def __call__(self, text):
        """Convert a Stanza Doc to a spaCy Doc.

        text (unicode): The text to process.
        RETURNS (spacy.tokens.Doc): The spaCy Doc object.
        """
        snlp_doc = self.snlp(text) if text else Document("")
        text = snlp_doc.text
        tokens, heads = self.get_tokens_with_heads(snlp_doc)
        if not len(tokens):
            return Doc(self.vocab)
        words = []
        spaces = []
        pos = []
        tags = []
        deps = []
        lemmas = []
        offset = 0
        is_aligned = self.check_aligned(text, tokens)
        if not is_aligned:
            warnings.warn("Due to multiword token expansion, the original "
                "text has been replaced by space-separated expanded tokens.",
                stacklevel=4,
            )
        for i, token in enumerate(tokens):
            span = text[offset:]
            if is_aligned and not len(span):
                warnings.warn(
                    "Unable to align all tokens with the text. The doc may be "
                    "truncated.",
                    stacklevel=4,
                )
                break
            while len(span) and span[0].isspace():
                # If we encounter leading whitespace, skip one character ahead
                offset += 1
                span = text[offset:]
            words.append(token.text)
            # Make sure all strings are in the vocabulary
            pos.append(self.vocab.strings.add(token.upos or ""))
            tags.append(self.vocab.strings.add(token.xpos or token.feats or ""))
            deps.append(self.vocab.strings.add(token.deprel or ""))
            lemmas.append(self.vocab.strings.add(token.lemma or ""))
            offset += len(token.text)
            span = text[offset:]
            if i == len(tokens) - 1:
                spaces.append(False)
            elif not is_aligned:
                spaces.append(True)
            else:
                next_token = tokens[i + 1]
                spaces.append(not span.startswith(next_token.text))
        attrs = [POS, TAG, DEP, HEAD]
        array = numpy.array(list(zip(pos, tags, deps, heads)), dtype="uint64")
        doc = Doc(self.vocab, words=words, spaces=spaces).from_array(attrs, array)
        ents = []
        for ent in snlp_doc.entities:
            ent_span = doc.char_span(ent.start_char, ent.end_char, ent.type)
            ents.append(ent_span)
        if not all(ents):
            warnings.warn(
                f"Can't set named entities because the character offsets don't "
                f"map to valid tokens produced by the Stanza tokenizer:\n"
                f"Words: {words}\n"
                f"Entities: {[(e.text, e.type, e.start_char, e.end_char) for e in snlp_doc.entities]}",
                stacklevel=4,
            )
        else:
            doc.ents = ents
        # Overwrite lemmas separately to prevent them from being overwritten by spaCy
        lemma_array = numpy.array([[lemma] for lemma in lemmas], dtype="uint64")
        doc.from_array([LEMMA], lemma_array)
        if any(pos) and any(tags):
            doc.is_tagged = True
        if any(deps):
            doc.is_parsed = True
        return doc

    def pipe(self, texts):
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts.
        YIELDS (Doc): A sequence of Doc objects, in order.
        """
        for text in texts:
            yield self(text)

    def get_tokens_with_heads(self, snlp_doc):
        """Flatten the tokens in the Stanza Doc and extract the token indices
        of the sentence start tokens to set is_sent_start.

        snlp_doc (stanza.Document): The processed Stanza doc.
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
                    if word.head:
                        head = word.head + offset - len(tokens) - 1
                    else:
                        head = 0
                    heads.append(head)
                    tokens.append(word)
            offset += sum(len(token.words) for token in sentence.tokens)
        return tokens, heads

    def check_aligned(self, text, tokens):
        token_texts = "".join(t.text for t in tokens)
        return re.sub(self._ws_pattern, "", text) == token_texts
