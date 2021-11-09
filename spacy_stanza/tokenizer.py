import warnings
from typing import Optional, Union

from spacy.tokens import Doc
from spacy.util import registry

from stanza import Pipeline
from stanza.resources.common import DEFAULT_MODEL_DIR
from stanza.models.common.vocab import UNK_ID
from stanza.models.common.pretrain import Pretrain


@registry.tokenizers("spacy_stanza.PipelineAsTokenizer.v1")
def create_tokenizer(
    lang: str = "",
    dir: Optional[str] = None,
    package: str = "default",
    processors: Union[dict, str] = {},
    logging_level: Optional[Union[int, str]] = None,
    verbose: Optional[bool] = None,
    use_gpu: bool = True,
    kwargs: dict = {},
):
    def tokenizer_factory(
        nlp,
        lang=lang,
        dir=dir,
        package=package,
        processors=processors,
        logging_level=logging_level,
        verbose=verbose,
        use_gpu=use_gpu,
        kwargs=kwargs,
    ) -> StanzaTokenizer:
        if dir is None:
            dir = DEFAULT_MODEL_DIR
        snlp = Pipeline(
            lang=lang,
            dir=dir,
            package=package,
            processors=processors,
            logging_level=logging_level,
            verbose=verbose,
            use_gpu=use_gpu,
            **kwargs,
        )
        return StanzaTokenizer(
            snlp,
            nlp.vocab,
        )

    return tokenizer_factory


class StanzaTokenizer(object):
    """Because we're only running the Stanza pipeline once and don't split
    it up into spaCy pipeline components, we'll set all the attributes within
    a custom tokenizer.
    """

    def __init__(self, snlp, vocab):
        """Initialize the tokenizer.

        snlp (stanza.Pipeline): The initialized Stanza pipeline.
        vocab (spacy.vocab.Vocab): The vocabulary to use.
        RETURNS (Tokenizer): The custom tokenizer.
        """
        self.snlp = snlp
        self.vocab = vocab
        self.svecs = self._find_embeddings(snlp)

    def __call__(self, text):
        """Convert a Stanza Doc to a spaCy Doc.

        text (unicode): The text to process.
        RETURNS (spacy.tokens.Doc): The spaCy Doc object.
        """
        if not text:
            return Doc(self.vocab)
        elif text.isspace():
            return Doc(self.vocab, words=[text], spaces=[False])

        snlp_doc = self.snlp(text)
        text = snlp_doc.text
        snlp_tokens, snlp_heads = self.get_tokens_with_heads(snlp_doc)
        words = []
        spaces = []
        pos = []
        tags = []
        morphs = []
        deps = []
        heads = []
        lemmas = []
        offset = 0
        token_texts = [t.text for t in snlp_tokens]
        is_aligned = True
        try:
            words, spaces = self.get_words_and_spaces(token_texts, text)
        except ValueError:
            words = token_texts
            spaces = [True] * len(words)
            is_aligned = False
            warnings.warn(
                "Due to multiword token expansion or an alignment "
                "issue, the original text has been replaced by space-separated "
                "expanded tokens.",
                stacklevel=4,
            )
        offset = 0
        for i, word in enumerate(words):
            if word.isspace() and (
                i + offset >= len(snlp_tokens) or word != snlp_tokens[i + offset].text
            ):
                # insert a space token
                pos.append("SPACE")
                tags.append("_SP")
                morphs.append("")
                deps.append("")
                lemmas.append(word)

                # increment any heads left of this position that point beyond
                # this position to the right (already present in heads)
                for j in range(0, len(heads)):
                    if j + heads[j] >= i:
                        heads[j] += 1

                # decrement any heads right of this position that point beyond
                # this position to the left (yet to be added from snlp_heads)
                for j in range(i + offset, len(snlp_heads)):
                    if j + snlp_heads[j] < i + offset:
                        snlp_heads[j] -= 1

                # initial space tokens are attached to the following token,
                # otherwise attach to the preceding token
                if i == 0:
                    heads.append(1)
                else:
                    heads.append(-1)

                offset -= 1
            else:
                token = snlp_tokens[i + offset]
                assert word == token.text

                pos.append(token.upos or "")
                tags.append(token.xpos or token.upos or "")
                morphs.append(token.feats or "")
                deps.append(token.deprel or "")
                heads.append(snlp_heads[i + offset])
                lemmas.append(token.lemma or "")

        doc = Doc(
            self.vocab,
            words=words,
            spaces=spaces,
            pos=pos,
            tags=tags,
            morphs=morphs,
            lemmas=lemmas,
            deps=deps,
            heads=[head + i for i, head in enumerate(heads)],
        )
        ents = []
        for ent in snlp_doc.entities:
            ent_span = doc.char_span(ent.start_char, ent.end_char, ent.type)
            ents.append(ent_span)
        if not is_aligned or not all(ents):
            warnings.warn(
                f"Can't set named entities because of multi-word token "
                f"expansion or because the character offsets don't map to "
                f"valid tokens produced by the Stanza tokenizer:\n"
                f"Words: {words}\n"
                f"Entities: {[(e.text, e.type, e.start_char, e.end_char) for e in snlp_doc.entities]}",
                stacklevel=4,
            )
        else:
            doc.ents = ents

        if self.svecs is not None:
            doc.user_token_hooks["vector"] = self.token_vector
            doc.user_token_hooks["has_vector"] = self.token_has_vector
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

    def get_words_and_spaces(self, words, text):
        if "".join("".join(words).split()) != "".join(text.split()):
            raise ValueError("Unable to align mismatched text and words.")
        text_words = []
        text_spaces = []
        text_pos = 0
        # normalize words to remove all whitespace tokens
        norm_words = [word for word in words if not word.isspace()]
        # align words with text
        for word in norm_words:
            try:
                word_start = text[text_pos:].index(word)
            except ValueError:
                raise ValueError("Unable to align mismatched text and words.")
            if word_start > 0:
                text_words.append(text[text_pos : text_pos + word_start])
                text_spaces.append(False)
                text_pos += word_start
            text_words.append(word)
            text_spaces.append(False)
            text_pos += len(word)
            if text_pos < len(text) and text[text_pos] == " ":
                text_spaces[-1] = True
                text_pos += 1
        if text_pos < len(text):
            text_words.append(text[text_pos:])
            text_spaces.append(False)
        return (text_words, text_spaces)

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

    # dummy serialization methods
    def to_bytes(self, **kwargs):
        return b""

    def from_bytes(self, _bytes_data, **kwargs):
        return self

    def to_disk(self, _path, **kwargs):
        return None

    def from_disk(self, _path, **kwargs):
        return self
