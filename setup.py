from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='casymir',
    version='1.0.0',
    author='Gustavo Pacheco',
    author_email='gustavo.pachecoguevara@radboudumc.nl',
    description='Generalized cascaded linear system model for x-ray detectors',
    license='MIT',
    url='https://github.com/radboud-axti/casymir_public',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    keywords='x-ray, imaging, detectors, medical physics, cascaded model',
    install_requires=[
        'numpy',
        'scipy',
        'spekpy',
        'xraydb',
        'pyYAML'
    ],
    entry_points={
        'console_scripts': ['casymir = casymir.module:run_casymir']
    }
)
