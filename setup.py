from os import path
from setuptools import find_packages, setup
from mapping_cli import __version__

with open('requirements.txt') as f:
    required = f.read().splitlines()
curdir = path.abspath(path.dirname(__file__))
with open(path.join(curdir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
setup(
    name="hams",
    packages=find_packages(),
    version=__version__,
    license="GPLv3+", 
    description="HAMS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vaibhav Balloli, Anurag Ghosh, Harsh Vijay, Jonathan Samuel, Akshay Nambi",
    author_email="t-vballoli@microsoft.com", 
    install_requires=required,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)", 
        "Programming Language :: Python :: 3.8",
    ]
)