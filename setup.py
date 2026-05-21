from setuptools import setup, find_packages
import os

# README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(os.path.join("umapvipr", "__init__.py"), "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"\'')
            break
    else:
        version = "2.0.0"

setup(
    name="umapvipr",
    version=version,
    description="Umap Visual Processor - Converts H5AD files into web-friendly formats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scanpy>=1.9.0",
        "h5py>=3.0.0",
        "psutil>=5.8.0"
    ],
    entry_points={
        "console_scripts": [
            "umapvipr=umapvipr.cli:main"
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="bioinformatics single-cell h5ad spatial-transcriptomics",
)
