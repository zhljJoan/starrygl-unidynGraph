import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import starry_unigraph

project = 'StarryUniGraph'
copyright = '2023, StarryUniGraph Team'
author = 'StarryUniGraph Team'

version = getattr(starry_unigraph, '__version__', '0.1.0')
release = getattr(starry_unigraph, '__version__', '0.1.0')


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.viewcode",
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
