# NexusSim Software Manual — Sphinx Configuration
# =================================================

project = 'NexusSim'
author = 'NexusSim Development Team'
copyright = '2024–2026, NexusSim Development Team'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
]

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'substitution',
    'tasklist',
]

myst_heading_anchors = 4

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_title = 'NexusSim Software Manual'
html_static_path = []

html_theme_options = {
    'repository_url': '',
    'use_repository_button': False,
    'show_toc_level': 3,
    'navigation_with_keys': True,
    'show_navbar_depth': 2,
}

# Number figures, tables, and code blocks
numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
}

# -- Options for LaTeX / PDF output ------------------------------------------

latex_engine = 'pdflatex'

latex_documents = [
    (master_doc, 'NexusSim_Software_Manual.tex',
     'NexusSim Software Manual',
     'NexusSim Development Team', 'manual'),
]

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{enumitem}
\setlistdepth{9}

% Custom title page
\renewcommand{\sphinxmaketitle}{%
  \begin{titlepage}
    \vspace*{3cm}
    \begin{center}
      {\Huge\bfseries NexusSim Software Manual\par}
      \vspace{1cm}
      {\Large Version 1.0.0\par}
      \vspace{2cm}
      {\large A Multi-Physics Computational Mechanics Framework\par}
      \vspace{0.5cm}
      {\large C++20 / Kokkos\par}
      \vspace{3cm}
      {\large NexusSim Development Team\par}
      \vspace{1cm}
      {\large March 2026\par}
      \vspace{2cm}
      {\small\textcopyright\ 2024--2026 NexusSim Development Team.
       Licensed under the Apache License, Version 2.0.\par}
    \end{center}
  \end{titlepage}
}
''',
    'extraclassoptions': 'openany,oneside',
    'figure_align': 'htbp',
    'fncychap': r'\usepackage[Sonny]{fncychap}',
}

latex_show_urls = 'footnote'
latex_show_pagerefs = True
