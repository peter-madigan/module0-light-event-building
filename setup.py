from setuptools import setup, find_packages
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
        name='module0-light-event-file',
        version='0.0',
        description='',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/peter-madigan/module0_light_event_file',
        author='Peter Madigan',
        author_email='pmadigan@lbl.gov',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3'
        ],
        packages=find_packages(),
        install_requires=[
            'h5py',
            'numpy',
            'ROOT',
            'root_numpy',
            'tqdm'
            ],
        scripts=[
            'light_event_builder.py'
            ],
)
