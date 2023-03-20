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
# sys.path.insert(0, os.path.abspath('~/Microsoft/OLA-Alt/HAMSApp/'))
# sys.path.insert(1, os.path.abspath('../'))
# sys.path.append(os.path.abspath('/Users/balli/Microsoft/'))
# sys.path.append(os.path.abspath('/Users/balli/Microsoft/Ola-ALT/'))
# sys.path.append(os.path.abspath('/Users/balli/Microsoft/Ola-ALT/HAMSApp/'))
# sys.path.append(os.path.abspath('/Users/balli/Microsoft/Ola-ALT/HAMSApp/ALT/'))
sys.path.append('../')
# sys.path.insert(2, os.path.abspath('../../'))
# sys.path.insert(3, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'HAMS'
copyright = '2023, Microsoft Research'
author = 'Anurag Ghosh, Harsh Vijay, Vaibhav Balloli, Jonathan Samuel, Akshay Nambi'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinxemoji.sphinxemoji",
    "breathe",
    "sphinx_markdown_builder",
    "sphinx_copybutton",
    "jupyter_sphinx",
    # "myst_nb",
    "myst_parser",
    "sphinx_proof",
    "sphinx_design",
    "sphinxcontrib.video",
    "sphinx_togglebutton",
    "sphinx_tabs.tabs",
    "nbsphinx",
    "sphinxcontrib.youtube",
]
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": False,
}
autodoc_inherit_docstrings = True
myst_enable_extensions = ["colon_fence"]

nb_execution_mode = "off"
nbsphinx_allow_errors = True
nbsphinx_execute = "never"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme_options = {
    "repository_url": "https://github.com/microsoft/HAMS",
    "use_repository_button": True,
    "use_download_button": True,
}

html_title = "HAMS Docs"
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "furo"
html_theme = "sphinx_book_theme"

# removes the .txt suffix
html_sourcelink_suffix = ""


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]
# source_suffix = ['.rst', '.md']

autodoc_mock_imports = ['alabaster', 'amqp', 'anacondaclient', 'anacondaproject', 'anyjson', 'apscheduler', 'asncrypto', 'astroid', 'astropy', 'atomicwrites', 'attrs', 'autobahn', 'automat', 'babel', 'backcall', 'backportsos', 'backportsshutilgetterminalsize', 'bcrypt', 'beautifulsoup', 'billiard', 'bitarray', 'bkcharts', 'bleach', 'bokeh', 'boto', 'bottleneck', 'celery', 'celluloid', 'certifi', 'cffi', 'chardet', 'click', 'cloudpickle', 'clyent', 'cognitiveface', 'colorama', 'comtypes', 'constantly', 'contextlib', 'cryptography', 'cycler', 'cython', 'cytoolz', 'dask', 'decorator', 'defusedxml', 'distributed', 'dlib', 'dnspython', 'docutils', 'entrypoints', 'etxmlfile', 'eventlet', 'fastcache', 'filelock', 'filterpy', 'flask', 'flaskbcrypt', 'flasklogin', 'flaskmarshmallow', 'flasksession', 'flasksqlacodegen', 'flasksqlalchemy', 'flaskwkhtmltopdf', 'gevent', 'greenlet', 'hpy', 'haversine', 'heapdict', 'hexdump', 'hkdf', 'htmllib', 'humanize', 'hyperlink', 'idna', 'imageio', 'imagesize', 'importlibmetadata', 'incremental', 'inflect', 'ipykernel', 'ipython', 'ipythongenutils', 'ipywidgets', 'isort', 'itsdangerous', 'jdcal', 'jedi', 'jinja', 'jsonschema', 'jupyter', 'jupyterclient', 'jupyterconsole', 'jupytercore', 'jupyterlab', 'jupyterlabserver', 'keyring', 'kiwisolver', 'kombu', 'laika', 'lazyobjectproxy', 'libusb', 'llvmlite', 'locket', 'lxml', 'markupsafe', 'marshmallow', 'marshmallowsqlalchemy', 'matplotlib', 'mccabe', 'menuinst', 'mistune', 'mklfft', 'mklrandom', 'monotonic', 'moreitertools', 'mpmath', 'msgpack', 'multipledispatch', 'mysqlconnector', 'nbconvert', 'nbformat', 'networkx', 'nltk', 'nose', 'notebook', 'numba', 'numexpr', 'numpy', 'numpydoc', 'olefile', 'opencvcontribpython', 'openpyxl', 'packaging', 'pandas', 'pandocfilters', 'parso', 'partd', 'pathpy', 'pathlib', 'patsy', 'pdfkit', 'pep', 'pickleshare', 'pillow', 'pip', 'pluggy', 'ply', 'prometheusclient', 'prompttoolkit', 'psutil', 'py', 'pycodestyle', 'pycosat', 'pycparser', 'pycrypto', 'pycurl', 'pyflakes', 'pygments', 'pyhamcrest', 'pylint', 'pymysql', 'pynacl', 'pyodbc', 'pyopenssl', 'pyparsing', 'pyquaternion', 'pyreadline', 'pyrsistent', 'pysocks', 'pytest', 'pytestarraydiff', 'pytestastropy', 'pytestdoctestplus', 'pytestopenfiles', 'pytestremotedata', 'pythondateutil', 'pytz', 'pywavelets', 'pywin', 'pywinctypes', 'pywinpty', 'pyyaml', 'pyzmq', 'qtawesome', 'qtconsole', 'qtpy', 'redis', 'requests', 'rope', 'ruamelyaml', 'scikitimage', 'scikitlearn', 'scipy', 'seaborn', 'sendtrash', 'setuptools', 'shapelypost', 'simplegeneric', 'singledispatch', 'six', 'snowballstemmer', 'sortedcollections', 'sortedcontainers', 'soupsieve', 'sphinx', 'sphinxcontribwebsupport', 'spyder', 'spyderkernels', 'sqlalchemy', 'statsmodels', 'sympy', 'tables', 'tblib', 'terminado', 'testpath', 'toolz', 'tornado', 'tqdm', 'traitlets', 'txaio', 'typedast', 'tzlocal', 'unicodecsv', 'urllib', 'vine', 'wcwidth', 'webencodings', 'werkzeug', 'wfastcgi', 'wheel', 'widgetsnbextension', 'wincertstore', 'wininetpton', 'winunicodeconsole', 'wrapt', 'xlrd', 'xlsxwriter', 'xlwings', 'xlwt', 'zict', 'zipp', 'zopeinterface', 'mysql']

autodoc_mock_imports += ['ffmpeg', 'flask_sqlalchemy', 'flask_bcrypt', 'flask_cors', 'flask_session', 'sqlalchemy_utils', 'cv2', 'PIL', 'torch', 'torchvision', 'reporter', 'cognitive_face', 'shapely', 'sklearn', 'moviepy', 'natsort' ,'decord', 'hydra', 'typer', 'omegaconf']