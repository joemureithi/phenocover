#!/usr/bin/env python3
"""
Setup script for PhenoCover - Weather-Enhanced Wheat Phenology Analysis Tool
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(
    encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip()
                        and not line.startswith("#")]
else:
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "requests>=2.25.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
    ]

setup(
    name="phenocover",
    version="0.1.0",
    author="Joseph Gitahi",
    author_email="joemureithi@live.com",
    description="Weather-Enhanced Wheat Phenology and Ground Cover Estimation Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joemureithi/phenocover",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="phenology agriculture wheat NDVI vegetation remote-sensing weather GDD",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phenocover=phenocover.cli:app",
        ],
    },
    include_package_data=True,
    package_data={
        "phenocover": [
            "docs/*.md",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/joemureithi/phenocover/issues",
        "Source": "https://github.com/joemureithi/phenocover",
        "Documentation": "https://github.com/joemureithi/phenocover/blob/main/README.md",
    },
    zip_safe=False,
)
