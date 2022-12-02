#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# libCEED documentation build configuration file, created by
# sphinx-quickstart on Tue Jan  7 18:59:28 2020.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import glob
import shutil
import sys
import breathe
import os
import subprocess
from sphinxcontrib import katex

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'altair.sphinxext.altairplot',
    'breathe',
    'hoverxref.extension',
    'sphinx_panels',
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.ifconfig',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinxcontrib.katex',
    'sphinxcontrib.mermaid',  # still in beta; fails with latexpdf builder
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.rsvgconverter',
]

# The following, if true, allows figures, tables and code-blocks to be
# automatically numbered if they have a caption.
numfig = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'libCEED'
copyright = '2020, LLNL, University of Colorado, University of Illinois, University of Tennesee, and the authors'
with open('../../../AUTHORS') as f:
    authorlist = f.readlines()
author = ', '.join(authorlist)

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
with open('../../../ceed.pc.template') as f:
    pkgconf_version = 'unknown'
    for line in f:
        if line.startswith('Version:'):
            pkgconf_version = line.partition(': ')[2]
            break
version = pkgconf_version
# The full version, including alpha/beta/rc tags.
release = pkgconf_version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    'examples/README.md',
    'examples/ceed/README.md',
    'examples/fluids/README.md',
    'examples/nek/README.md',
    'examples/petsc/README.md',
    'examples/solid/README.md',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# sphinxcontrib-bibtex 2.0 requires listing all bibtex files here
bibtex_bibfiles = [
    'references.bib',
]

myst_enable_extensions = [
    'deflist',
    'dollarmath',
    'html_image',
    'linkify',
    'colon_fence',
]

myst_heading_anchors = 2
myst_url_schemes = ["http", "https", "mailto"]

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["css"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# hoverxref options
hoverxref_auto_ref = True
hoverxref_mathjax = True
hoverxref_role_types = {
    'ref': 'modal',
}

latex_macros = r"""
\def \diff {\operatorname{d}\!}
\def \tcolon {\!:\!}
\def \trace {\operatorname{trace}}
"""

# Translate LaTeX macros to KaTeX and add to options for HTML builder
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = 'macros: {' + katex_macros + '}'

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'libCEEDdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_engine = 'xelatex'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    'preamble': r"""
\usepackage{bm}
\usepackage{amscd}
""" + latex_macros,

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    'fontenc': r'\usepackage{mathspec}',
    'fontpkg': r"""
\setmainfont{TeX Gyre Pagella}
\setmathfont{TeX Gyre Pagella Math}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
""",
}

latex_logo = '../../img/ceed-full-name-logo.PNG'

latexauthorslist = r' \and '.join(authorlist)

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'libCEED.tex', 'libCEED User Manual',
     latexauthorslist, 'howto'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'libceed', 'libCEED User Manual',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'libCEED', 'libCEED User Manual',
     latexauthorslist, 'libCEED', 'Efficient implementations of finite element operators.',
     'Miscellaneous'),
]


# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
}


# -- Options for breathe --------------------------------------------------
sys.path.append(breathe.__path__)
breathe_projects = {"libCEED": "../../../xml"}
breathe_default_project = "libCEED"
breathe_build_directory = "../build/breathe"
breathe_domain_by_extension = {"c": "c", "h": "c", "cpp": "cpp", "hpp": "cpp"}

# Run Doxygen if building on Read The Docs
rootdir = os.path.join(os.path.dirname(__file__),
                       os.pardir, os.pardir, os.pardir)
if os.environ.get('READTHEDOCS'):
    subprocess.check_call(['doxygen', 'Doxyfile'], cwd=rootdir)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


# Copy example documentation from source tree
try:
    shutil.rmtree('examples')
except FileNotFoundError:
    pass
for filename in glob.glob(os.path.join(
        rootdir, 'examples/**/*.md'), recursive=True):
    destdir = os.path.dirname(os.path.relpath(filename, rootdir))
    mkdir_p(destdir)
    shutil.copy2(filename, destdir)
shutil.copy2(os.path.join(rootdir, 'README.md'), '.')

for filename in glob.glob(os.path.join(
        rootdir, 'examples/**/*.csv'), recursive=True):
    destdir = os.path.dirname(os.path.relpath(filename, rootdir))
    mkdir_p(destdir)
    shutil.copy2(filename, destdir)
