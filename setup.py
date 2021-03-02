import os
from setuptools import setup, find_packages

VERSION = '0.0.01'

# ml, short for `metric learning`
MODULE_NAME = 'pyhandle'
AUTHORS = 'Sixigma'


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


install_requires = [
    'torch',
    'numpy',
    'torchvision'
]


setup_info = dict(
    name='pytorch handle',
    author=AUTHORS,
    version=VERSION,
    description='Pytorch applications',
    long_discription=read('README.md'),
    license='BSD',
    url='https://github.com/Kuanch/pyhandle',
    install_requires=install_requires,
    include_package_data=True,
    packages=find_packages()
)
# Install evaluation
setup(**setup_info)
