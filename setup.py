from setuptools import setup, find_packages

setup(
    name='casymir',
    version='0.1.0a1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'spekpy',
        'xraydb',
        'pyYAML'
    ],
    entry_points={
        'console_scripts': ['casymir = casymir.module:example']
    }
)
