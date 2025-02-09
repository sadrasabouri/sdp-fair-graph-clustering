# -*- coding: utf-8 -*-
"""Setup module."""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def get_requires():
    """Read requirements.txt."""
    requirements = open("requirements.txt", "r").read()
    return list(filter(lambda x: x != "", requirements.split()))


def read_description():
    """Read README.md and CHANGELOG.md."""
    try:
        with open("README.md") as r:
            description = "\n"
            description += r.read()
        with open("CHANGELOG.md") as c:
            description += "\n"
            description += c.read()
        return description
    except Exception:
        return '''A Semidefinite Relaxation Approach for Fair Graph Clustering'''


setup(
    name='srfsc',
    packages=[
        'srfsc', ],
    version='0.1',
    description='A Semidefinite Relaxation Approach for Fair Graph Clustering',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    install_requires=get_requires(),
    include_package_data=True,
    python_requires='>=3.8',
    license='MIT',
)
