# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hiaerspike-web'
copyright = '2025, Gwen Frank'
author = 'Gwen Frank'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
]

myst_enable_extensions = ["dollarmath"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'custom.css',  # Path relative to the _static directory
]
html_theme_options = {
    "show_prev_next": False,
    "logo": {
        "image_light": "logo.svg",
        "image_dark": "logo_dark.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Integrated-Systems-Neuroengineering/hs_api",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Bluesky",
            "url": "https://bsky.app/profile/hiaer-spike.bsky.social",
            "icon": "fa-brands fa-bluesky",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"]
# ...
}


sphinx_gallery_conf = {
     'examples_dirs': '../../webexamples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
