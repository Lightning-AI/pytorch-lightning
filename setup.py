#!/usr/bin/env python

from setuptools import setup, find_packages, os

# https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open(os.path.join("src", "pytorch-lightning", "__init__.py")) as fp:
    exec(fp.read(), version)

# http://blog.ionelmc.ro/2014/05/25/python-packaging/
setup(
    name="pytorch-lightning",
    version=version["__version__"],
    description="The Keras for ML researchers using PyTorch",
    author="William Falcon",
    author_email="waf2107@columbia.edu",
    url="https://github.com/williamFalcon/pytorch-lightning",
    download_url="https://github.com/williamFalcon/pytorch-lightning",
    license="MIT",
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.5",
    install_requires=[
        "torch",
        "tqdm",
        "test-tube",
    ],
    extras_require={
        "dev": [
            "black ; python_version>='3.6'",
            "coverage",
            "isort",
            "pytest",
            "pytest-cov<2.6.0",
            "pycodestyle",
            "sphinx",
            "nbsphinx",
            "ipython>=5.0",
            "jupyter-client",
        ]
    },
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["pytorch-lightning=pytorch-lightning.cli:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    long_description=open("README.md", encoding="utf-8").read(),
    include_package_data=True,
    zip_safe=False,
)
