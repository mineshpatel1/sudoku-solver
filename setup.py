import setuptools
import os

with open('README.md', 'r') as fh:
    long_description = fh.read()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('sudoku_solver/data')


setuptools.setup(
    name='mineshpatel1_sudoku_solver',
    version='0.1',
    description='sudoku parser & solver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mineshpatel1/sudoku-solver',
    packages=['sudoku_solver'],
    package_data={'':extra_files},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=['opencv-python', 'tensorflow', 'matplotlib'],
    python_requires='>=3.8')
