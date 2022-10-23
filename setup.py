import os
import setuptools

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    README = readme.read()

setuptools.setup(
    name="batch_checkpoint_merger",
    version="0.0.3",
    author="lodimas",
    author_email="lodimas123@gmail.com",
    description="Batch checkpoint merger, with a UI",
    long_description="Batch checkpoint merger for merging stable diffusion models en masse",
    long_description_content_type="text/markdown",
    url="https://github.com/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['batch-chk-mrg=batch_checkpoint_merger.batch_checkpoint_merger:main'],
    },
    install_requires=[
        'PySimpleGUI',
        'matplotlib',
        'pathlib',
        'torch',
        'pyperclip'
    ]
)
