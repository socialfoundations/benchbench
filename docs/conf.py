# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

project = 'BenchBench'
copyright = '2024, Guanhua'
author = 'Guanhua'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # pull doc from docstrings
    'sphinx.ext.intersphinx',  # link to other projects
    'sphinx.ext.todo',  # support TODOs
    'sphinx.ext.ifconfig',  # include stuff based on configuration
    'sphinx.ext.viewcode',  # add source code
    'myst_parser',  # add MD files
    'sphinx.ext.napoleon'  # Google style doc
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
