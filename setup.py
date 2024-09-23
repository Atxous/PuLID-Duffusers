from setuptools import setup, find_packages

import os

def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

setup(
    name='pulid_pipelines',
    version='0.1-beta',
    packages=find_packages(),
    description='Easily integrate PuLID with Hugging Face pipelines for identity customization in image generation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SmokDev/PuLID-Pipelines',
    author='SmokDev',
    classifiers=[
        'Development Status :: 0.1 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
    install_requires=get_requirements(),
)