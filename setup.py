#!/usr/bin/env python
from __future__ import unicode_literals

import os
import io
from setuptools import setup, find_packages


def setup_package():
    package_name = "spacy_stanza"
    root = os.path.abspath(os.path.dirname(__file__))

    # Read in package meta from about.py
    about_path = os.path.join(root, package_name, "about.py")
    with io.open(about_path, encoding="utf8") as f:
        about = {}
        exec(f.read(), about)

    # Get readme
    readme_path = os.path.join(root, "README.md")
    with io.open(readme_path, encoding="utf8") as f:
        readme = f.read()

    setup(
        name="spacy-stanza",
        description=about["__summary__"],
        long_description=readme,
        long_description_content_type="text/markdown",
        author=about["__author__"],
        author_email=about["__email__"],
        url=about["__uri__"],
        version=about["__version__"],
        license=about["__license__"],
        packages=find_packages(),
        install_requires=["spacy>=2.1.0,<3.0.0", "stanza>=1.0.0,<1.3.0"],
        python_requires=">=3.6",
        entry_points={
            "spacy_languages": [
                "stanza_af = spacy_stanza:StanzaLanguage",
                "stanza_ar = spacy_stanza:StanzaLanguage",
                "stanza_bg = spacy_stanza:StanzaLanguage",
                "stanza_ca = spacy_stanza:StanzaLanguage",
                "stanza_cs = spacy_stanza:StanzaLanguage",
                "stanza_da = spacy_stanza:StanzaLanguage",
                "stanza_de = spacy_stanza:StanzaLanguage",
                "stanza_el = spacy_stanza:StanzaLanguage",
                "stanza_en = spacy_stanza:StanzaLanguage",
                "stanza_es = spacy_stanza:StanzaLanguage",
                "stanza_et = spacy_stanza:StanzaLanguage",
                "stanza_eu = spacy_stanza:StanzaLanguage",
                "stanza_fa = spacy_stanza:StanzaLanguage",
                "stanza_fi = spacy_stanza:StanzaLanguage",
                "stanza_fr = spacy_stanza:StanzaLanguage",
                "stanza_ga = spacy_stanza:StanzaLanguage",
                "stanza_he = spacy_stanza:StanzaLanguage",
                "stanza_hi = spacy_stanza:StanzaLanguage",
                "stanza_hr = spacy_stanza:StanzaLanguage",
                "stanza_hu = spacy_stanza:StanzaLanguage",
                "stanza_hy = spacy_stanza:StanzaLanguage",
                "stanza_id = spacy_stanza:StanzaLanguage",
                "stanza_it = spacy_stanza:StanzaLanguage",
                "stanza_ja = spacy_stanza:StanzaLanguage",
                "stanza_ko = spacy_stanza:StanzaLanguage",
                "stanza_lt = spacy_stanza:StanzaLanguage",
                "stanza_lv = spacy_stanza:StanzaLanguage",
                "stanza_mr = spacy_stanza:StanzaLanguage",
                "stanza_nb = spacy_stanza:StanzaLanguage",
                "stanza_nl = spacy_stanza:StanzaLanguage",
                "stanza_pl = spacy_stanza:StanzaLanguage",
                "stanza_pt = spacy_stanza:StanzaLanguage",
                "stanza_ro = spacy_stanza:StanzaLanguage",
                "stanza_ru = spacy_stanza:StanzaLanguage",
                "stanza_sk = spacy_stanza:StanzaLanguage",
                "stanza_sl = spacy_stanza:StanzaLanguage",
                "stanza_sr = spacy_stanza:StanzaLanguage",
                "stanza_sv = spacy_stanza:StanzaLanguage",
                "stanza_ta = spacy_stanza:StanzaLanguage",
                "stanza_te = spacy_stanza:StanzaLanguage",
                "stanza_tr = spacy_stanza:StanzaLanguage",
                "stanza_uk = spacy_stanza:StanzaLanguage",
                "stanza_ur = spacy_stanza:StanzaLanguage",
                "stanza_vi = spacy_stanza:StanzaLanguage",
                "stanza_zh = spacy_stanza:StanzaLanguage",
            ]
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
