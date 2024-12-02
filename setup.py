from setuptools import setup, find_packages

setup(
    name='constrained_learning',
    version='3.1',
    packages=find_packages(),
    license=None,
    description='Handy package for training output constrained neural networks',
    long_description=open('README.md').read(),
    install_requires=[
        "cvxopt>=1.3.1",
        "dill>=0.3.6",
        "pynput",
        "matplotlib>=3.7.1",
        "numpy",
        "pyLasaDataset>=0.1.1",
        "scikit_learn>=1.2.2",
        "scipy>=1.10.1",
        "torch>=2.0.1",
        "jupyterlab"
    ],
    url='https://github.com/ubi-coro/constrained_learning',
    author='Jannick Stranghöner',
    author_email='jannick.stranghoener@uni-bielefeld.de'
)
