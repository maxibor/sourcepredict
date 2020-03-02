import pathlib
from sourcepredict import __version__
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='sourcepredict',
    version=__version__,
    description='Classification and prediction of the origin of metagenomic samples',
    long_description=README,
    url='https://github.com/maxibor/sourcepredict',
    long_description_content_type="text/markdown",
    license='GPLv3',
    python_requires=">=3.6",
    install_requires=[
        'numpy >=1.16.4',
        'pandas >=0.24.1',
        'scikit-learn >=0.20.1',
        'scikit-bio >=0.5.5',
        'umap-learn >=0.3.7',
        'scipy >=1.1.0',
        'ete3 >=3.1.1'
    ],
    packages=find_packages(include=['sourcepredict', 'sourcepredict.sourcepredictlib']),
    entry_points={
        'console_scripts': [
            'sourcepredict= sourcepredict.__main__:main'
        ]
    }
)