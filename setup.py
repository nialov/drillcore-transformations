#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pip', 'Click', 'numpy', 'pandas', 'matplotlib', 'sympy', 'numpy', 'xlrd', 'openpyxl']

setup_requirements = []

test_requirements = ['pytest', 'hypothesis', 'numpy', 'yamllint', 'sphinx', 'tox', 'coverage', 'pytest-cov', 'guzzle-sphinx-theme', 'sphinx']

setup(
    author="Nikolas Ovaskainen",
    author_email='nikolasovaskainen@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    description="Drillcore Transformations allows for alpha, beta and gamma drillcore transformations.",
    entry_points={
        'console_scripts': [
            'drillcore_transformations=drillcore_transformations.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='drillcore_transformations',
    name='drillcore_transformations',
    packages=find_packages(include=['drillcore_transformations', 'drillcore_transformations.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nialov/drillcore-transformations',
    version='0.1.0',
    zip_safe=False,
)
