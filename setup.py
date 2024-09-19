from setuptools import setup, find_packages

setup(
    name='pulid-pipelines',
    version='0.1-beta',
    packages=find_packages(),
    package_dir={"": "pulid"},
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
)