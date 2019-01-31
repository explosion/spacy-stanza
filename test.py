# coding: utf8
from spacy_stanfordnlp import StanfordNLPLanguage
import stanfordnlp
import pytest


@pytest.fixture
def models_dir():
    return "./models"


@pytest.fixture
def lang():
    return "en"


def test_spacy_stanfordnlp(lang, models_dir):
    snlp = stanfordnlp.Pipeline(lang=lang, models_dir=models_dir)
    nlp = StanfordNLPLanguage(snlp)
    assert nlp.lang == lang

    doc = nlp("Hello world! This is a test.")

    # fmt: off
    assert [t.text for t in doc] == ["Hello", "world", "!", "This", "is", "a", "test", "."]
    assert [t.lemma_ for t in doc] == ["hello", "world", "!", "this", "be", "a", "test", "."]
    assert [t.pos_ for t in doc] == ["INTJ", "NOUN", "PUNCT", "DET", "VERB", "DET", "NOUN", "PUNCT"]
    assert [t.tag_ for t in doc] == ["UH", "NN", ".", "DT", "VBZ", "DT", "NN", '.']
    assert [t.dep_ for t in doc] == ["root", "vocative", "punct", "nsubj", "cop", "det", "root", "punct"]
    assert [t.is_sent_start for t in doc] == [True, None, None, True, None, None, None, None]
    assert [t.is_stop for t in doc] == [False, False, False, True, True, True, False, False]
    assert len(list(doc.sents)) == 2
    assert doc.is_tagged
    assert doc.is_parsed
    assert doc.is_sentenced
