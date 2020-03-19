import os.path as osp

import setuptools

import torchfurnace

this_directory = osp.abspath(osp.dirname(__file__))


def read_file(filename):
    with open(osp.join(this_directory, filename), 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


setuptools.setup(
    name=torchfurnace.__name__,
    python_requires='>=3.6',
    version=torchfurnace.__version__,
    author=torchfurnace.__author__,
    author_email=torchfurnace.__contact__,
    maintainer=torchfurnace.__author__,
    maintainer_email=torchfurnace.__contact__,
    description='A tool package for training model, pre-processing dataset and managing experiment record in pytorch AI tasks.',
    license='MIT License',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/tianyu-su/torchfurnace',
    packages=setuptools.find_packages(exclude=('tests', 'tests.*')),
    include_package_data=True,
    platforms=['any'],
    keywords=['pytorch', 'tracer', 'engine', 'torchfurnace'],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=read_file('requirements.txt')
)
