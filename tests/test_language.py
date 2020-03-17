# coding: utf8
from spacy_stanza.language import StanzaLanguage, get_defaults
from spacy.lang.en import EnglishDefaults
from spacy.language import BaseDefaults
import stanza
import pytest


@pytest.fixture
def lang():
    return "en"


def tags_equal(act, exp):
    """Check if each actual tag in act is equal to one or more expected tags in exp."""
    return all(a == e if isinstance(e, str) else a in e for a, e in zip(act, exp))


def test_spacy_stanza(lang):
    stanza.download(lang)
    snlp = stanza.Pipeline(lang=lang)
    nlp = StanzaLanguage(snlp)
    assert nlp.lang == "stanza_" + lang

    doc = nlp("Hello world! This is a test.")

    # Expected POS tags. Note: Different versions of stanza result in different POS tags.
    # In particular, "this" can be a determiner or pronoun, their distinction is pretty vague in
    # general. And "is" in "This is a test" can be interpreted either as simply a verb or as an
    # auxiliary (linking) verb. Neither interpretation is necessarily more or less correct.
    # fmt: off
    pos_exp = ["INTJ", "NOUN", "PUNCT", ("DET", "PRON"), ("VERB", "AUX"), "DET", "NOUN", "PUNCT"]

    assert [t.text for t in doc] == ["Hello", "world", "!", "This", "is", "a", "test", "."]
    assert [t.lemma_ for t in doc] == ["hello", "world", "!", "this", "be", "a", "test", "."]
    assert tags_equal([t.pos_ for t in doc], pos_exp)

    assert [t.tag_ for t in doc] == ["UH", "NN", ".", "DT", "VBZ", "DT", "NN", '.']
    assert [t.dep_ for t in doc] == ["root", "vocative", "punct", "nsubj", "cop", "det", "root", "punct"]
    assert [t.is_sent_start for t in doc] == [True, None, None, True, None, None, None, None]
    assert any([t.is_stop for t in doc])
    # fmt: on
    assert len(list(doc.sents)) == 2
    assert doc.is_tagged
    assert doc.is_parsed
    assert doc.is_sentenced

    docs = list(nlp.pipe(["Hello world", "This is a test"]))
    assert docs[0].text == "Hello world"
    assert [t.pos_ for t in docs[0]] == ["INTJ", "NOUN"]
    assert docs[1].text == "This is a test"
    assert tags_equal([t.pos_ for t in docs[1]], pos_exp[3:-1])
    assert doc.ents == tuple()

    # Test NER
    doc = nlp("Barack Obama was born in Hawaii.")
    assert len(doc.ents) == 2
    assert doc.ents[0].text == "Barack Obama"
    assert doc.ents[0].label_ == "PERSON"
    assert doc.ents[1].text == "Hawaii"
    assert doc.ents[1].label_ == "GPE"


def test_get_defaults():
    assert get_defaults("en") == EnglishDefaults
    assert get_defaults("xvkfokdfo") == BaseDefaults
