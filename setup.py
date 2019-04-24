#!/usr/bin/env python
from __future__ import unicode_literals

import os
import io
from setuptools import setup, find_packages


def setup_package():
    package_name = "spacy_stanfordnlp"
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
        name=about["__title__"],
        description=about["__summary__"],
        long_description=readme,
        long_description_content_type="text/markdown",
        author=about["__author__"],
        author_email=about["__email__"],
        url=about["__uri__"],
        version=about["__version__"],
        license=about["__license__"],
        packages=find_packages(),
        install_requires=["spacy>=2.1.0", "stanfordnlp>=0.1.0,<0.2.0"],
        python_requires=">=3.6",
        entry_points={
            "spacy_languages": [
                "stanfordnlp_en = spacy_stanfordnlp:StanfordNLPLanguage"
            ]
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
