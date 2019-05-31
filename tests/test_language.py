# coding: utf8
from spacy_stanfordnlp.language import StanfordNLPLanguage, get_defaults
from spacy.lang.en import EnglishDefaults
from spacy.language import BaseDefaults
import stanfordnlp
import pytest


@pytest.fixture
def models_dir():
    return "./models"


@pytest.fixture
def lang():
    return "en"


def tags_equal(act, exp):
    """Check if each actual tag in act is equal to one or more expected tags in exp."""
    return all(a == e if isinstance(e, str) else a in e for a, e in zip(act, exp))


def test_spacy_stanfordnlp(lang, models_dir):
    try:
        snlp = stanfordnlp.Pipeline(lang=lang, models_dir=models_dir)
    except:
        snlp = stanfordnlp.Pipeline(lang=lang)
    nlp = StanfordNLPLanguage(snlp)
    assert nlp.lang == "stanfordnlp_" + lang

    doc = nlp("Hello world! This is a test.")

    # Expected POS tags. Note: Different versions of stanfordnlp result in different POS tags.
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


def test_get_defaults():
    assert get_defaults("en") == EnglishDefaults
    assert get_defaults("xvkfokdfo") == BaseDefaults
