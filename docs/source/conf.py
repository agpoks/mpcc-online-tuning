import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "mpcc-online-tuning"
copyright = "2026, agpoks"
author = "agpoks"

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {"logo_only": True}
