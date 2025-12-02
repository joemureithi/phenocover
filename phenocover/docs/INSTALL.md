# PhenoCover Installation Guide

## Installation Methods

### Method 1: Install from Source (Recommended for Development)

1. **Clone the repository:**

```bash
git clone https://github.com/joemureithi/phenocover.git
cd phenocover
```

2. **Install in development mode:**

```bash
pip install -e .
```

This installs the package in editable mode, allowing you to modify the code and see changes immediately.

### Method 2: Install from Source (Standard Installation)

1. **Clone the repository:**

```bash
git clone https://github.com/joemureithi/phenocover.git
cd phenocover
```

2. **Install the package:**

```bash
pip install .
```

### Method 3: Install with Development Dependencies

For contributors who want to run tests and use development tools:

```bash
pip install -e ".[dev]"
```

This installs additional packages for:

- Testing (pytest, pytest-cov)
- Code formatting (black)
- Linting (flake8)
- Type checking (mypy)

## Verify Installation

After installation, verify that phenocover is properly installed:

```bash
# Check version
phenocover --version

# Get help
phenocover --help

# Generate sample config
phenocover generate-config --format yaml --output test_config.yml
```

## Quick Start After Installation

1. **Generate a configuration file:**

```bash
phenocover generate-config --format yaml --output my_config.yml
```

2. **Edit the configuration file** with your data paths

3. **Run analysis:**

```bash
phenocover phenology-analyzer --config my_config.yml
```

## Using as a Python Library

After installation, you can also import and use phenocover in your Python scripts:

```python
from phenocover.wheat_phenology_analyzer import WheatPhenologyAnalyzer

# Initialize analyzer
analyzer = WheatPhenologyAnalyzer(
    ndvi_file='path/to/ndvi_data.csv',
    sowing_date='03.10.2023',
    harvest_date='30.07.2024',
    geojson_file='path/to/field.geojson'
)

# Run analysis
analyzer.estimate_fvc_parameters(method='seasonal')
analyzer.interpolate_ndvi(method='balanced')
analyzer.estimate_growth_stages()
analyzer.create_visualization(save_path='results/plot.png')
analyzer.save_results(output_file='results/data.csv')
analyzer.generate_summary_report()
```

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Dependencies**: Automatically installed (see requirements.txt)

## Uninstallation

To uninstall phenocover:

```bash
pip uninstall phenocover
```

## Troubleshooting

### ImportError after installation

If you get import errors after installation, try:

```bash
pip install --force-reinstall phenocover
```

### Command not found: phenocover

Make sure your Python scripts directory is in your PATH:

- **Windows**: Usually `C:\Python3X\Scripts\`
- **Linux/Mac**: Usually `~/.local/bin` or `/usr/local/bin`

Or use the module directly:

```bash
python -m phenocover --help
```

### Missing dependencies

If you encounter missing dependencies:

```bash
pip install -r requirements.txt
```

## Building from Source

To build distribution packages:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# This creates:
# - dist/phenocover-0.1.0-py3-none-any.whl
# - dist/phenocover-0.1.0.tar.gz
```

## For Developers

### Setting up development environment

1. **Clone and install in development mode:**

```bash
git clone https://github.com/joemureithi/phenocover.git
cd phenocover
pip install -e ".[dev]"
```

2. **Run tests:**

```bash
pytest
```

3. **Format code:**

```bash
black phenocover/
```

4. **Check types:**

```bash
mypy phenocover/
```

## Support

For issues:

- GitHub Issues: <https://github.com/joemureithi/phenocover/issues>
