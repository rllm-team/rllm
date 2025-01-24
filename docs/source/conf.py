# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import rllm

sys.path.insert(0, os.path.abspath('../..'))



# -- Project information -----------------------------------------------------

project = 'rLLM'
copyright = '2024, rLLM Team'
author = 'Zheng Wang, Weichen Li, Xiaotong Huang, Enze Zhang'
version = '1.0'
# The full version, including alpha/beta/rc tags
# release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    # 'autoapi.extension',
    'sphinx.ext.autosummary',
    'sphinx_jinja',    
    "sphinx.ext.graphviz",
    "sphinxemoji.sphinxemoji",
    # "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "nbsphinx",
    "nbsphinx_link",

]

autosummary_generate = True
autosummary_imported_members = True

# autoapi_type = 'python'
# autoapi_dirs = ['../../rllm','../../examples']
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#3498db",  # 主题颜色 (蓝色)
        "color-brand-content": "#2c3e50",  # 内容颜色 (深蓝色)
        "color-background-primary": "#ffffff",  # 主背景颜色 (白色)
        "color-background-secondary": "#f2f2f2",  # 次背景颜色 (浅灰色)
        "color-sidebar-background": "ffffff",  # 侧边栏背景颜色 (白色)
        "color-sidebar-text": "#3498db",  # 侧边栏文本颜色 (蓝色)
    },
}


# 禁用页面源链接
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, 'templates'):
        rst_context = {'rllm': rllm}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    app.connect('source-read', rst_jinja_render)
