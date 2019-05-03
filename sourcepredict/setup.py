from setuptools import setup, find_packages
from codecs import open
from os import path
from sourcepredict.__main__ import __version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
print(install_requires)
dependency_links = [x.strip().replace('git+', '')
                    for x in all_reqs if x.startswith('git+')]

setup(
    name='sourcepredict_test',
    version=__version__,
    description='Prediction/source tracking of metagenomic samples source using machine learning',
    long_description=long_description,
    url='https://github.com/maxibor/sourcepredict',
    download_url='https://github.com/maxibor/sourcepredict/tarball/' + __version__,
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='Maxime Borry',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='borry@shh.mpg.de',
    entry_points={
        'console_scripts': [
            'sourcepredict=sourcepredict.__main__:main',
        ],
    },
)
