#!/usr/bin/env python

"""
The setup script for pip. Allows for `pip install -e .` installation.
"""

from setuptools import setup, find_packages

requirements = ['numpy', 'matplotlib', 'torch', 'PyYAML',
            'scikit-learn', 'pandas', 'cycler', 'tables']
setup_requirements = []
tests_requirements = ['pytest']

setup(
    author='L. Cheng',
    author_email='lionel.cheng@hotmail.fr',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8'
    ],
    description='pklotclass: ',
    install_requires=requirements,
    license='GNU General Public License v3',
    long_description='\n\n',
    include_package_data=True,
    keywords='parking classifier',
    name='pklotclass',
    packages=find_packages(include=['pklotclass']),
    setup_requires=setup_requirements,

    test_suite='tests',
    tests_require=tests_requirements,
    version='0.1',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'pklot_train=pklotclass.train:main',
            'pklot_trains=pklotclass.multiple_train:main',
            'pklot_eval=pklotclass.eval:main',
            'pklot_evals=pklotclass.multiple_eval:main',
            'pklot_plot=pklotclass.pproc:main',
            'pklot_predict=pklotclass.predict:main'
        ],
    },
)
