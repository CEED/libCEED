#!/usr/bin/env python
from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='junit-xml',
    author='Brian Beyer',
    author_email='brian@kyr.us',
    url='https://github.com/kyrus/python-junit-xml',
    license='MIT',
    packages=find_packages(),
    test_suite='test_junit_xml',
    description='Creates JUnit XML test result documents that can be read by '
                'tools such as Jenkins',
    long_description=read('README.rst'),
    version='1.8',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: Freely Distributable',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Build Tools',
        'Topic :: Software Development :: Testing',
        ],
    install_requires=[
        'six'
        ]
    )
