from setuptools import setup, find_packages


setup(
    packages=find_packages(),
    package_data={"eva_clip": ["bpe_simple_vocab_16e6.txt.gz"]},
)