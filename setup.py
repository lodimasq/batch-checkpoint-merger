import os
import setuptools

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    README = readme.read()

setuptools.setup(
    name="batch_checkpoint_merger",
    version="1.0.0",
    author="lodimas",
    author_email="lodimas123@gmail.com",
    description="Batch checkpoint merger, with a UI",
    long_description="Python based application to automate the creation of model checkpoint merges. Supports various interpolation models in an attempt to smooth the transition between merge steps.",
    long_description_content_type="text/markdown",
    url="https://github.com/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'PySimpleGUI',
        'matplotlib',
        'pathlib',
        'torch',
        'pyperclip'
    ]
)
