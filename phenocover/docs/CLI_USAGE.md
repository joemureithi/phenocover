# PhenoCover CLI Usage Guide

## Overview

PhenoCover provides a command-line interface for weather-enhanced wheat phenology analysis. The CLI supports parameter configuration via command-line options or configuration files (YAML/JSON).

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install typer rich pyyaml pandas numpy matplotlib seaborn scipy requests
```

## Commands

### 1. Generate Configuration File

Create a sample configuration file with default parameters:

```bash
# Generate YAML configuration (recommended)
python -m phenocover generate-config --format yaml --output config.yml

# Generate JSON configuration
python -m phenocover generate-config --format json --output config.json
```

**Options:**

- `--format, -f`: Configuration format (`yaml` or `json`, default: `yaml`)
- `--output, -o`: Output file path (default: `phenology_config.yaml` or `phenology_config.json`)

**Generated Configuration Structure:**

```yaml
ndvi_file: "NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv"
sowing_date: "03.10.2023"
harvest_date: "30.07.2024"
geojson_file: "field_location.geojson"
fvc_method: "seasonal"
interpolation_method: "balanced"
output_dir: "./results"
analysis_options:
  estimate_fvc: true
  interpolate_ndvi: true
  estimate_growth_stages: true
  create_visualization: true
  save_results: true
  generate_summary: true
weather_integration:
  enabled: true
  source: "Open-Meteo API"
```

### 2. Run Phenology Analysis

Run the phenology analysis using either a configuration file or command-line parameters.

#### Using Configuration File (Recommended)

```bash
python -m phenocover phenology-analyzer --config config.yml
```

#### Using Command-Line Parameters

```bash
python -m phenocover phenology-analyzer \
  --ndvi-file "NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv" \
  --sowing-date "03.10.2023" \
  --harvest-date "30.07.2024" \
  --geojson-file "field_location.geojson" \
  --fvc-method seasonal \
  --interpolation-method balanced
```

#### Hybrid Approach (Config + CLI Override)

Command-line parameters override configuration file values:

```bash
python -m phenocover phenology-analyzer \
  --config config.yml \
  --fvc-method custom \
  --output-dir ./custom_results
```

**Required Parameters:**

- `--ndvi-file, -n`: Path to NDVI CSV file
- `--sowing-date, -s`: Sowing date (format: DD.MM.YYYY)
- `--harvest-date, -h`: Harvest date (format: DD.MM.YYYY)
- `--geojson-file, -g`: Path to GeoJSON file with field location

**Optional Parameters:**

- `--config, -c`: Path to configuration file (YAML or JSON)
- `--fvc-method`: FVC estimation method (`seasonal`, `dimyuk`, or `custom`, default: `seasonal`)
- `--interpolation-method`: NDVI interpolation method (`balanced`, `linear`, or `cubic`, default: `balanced`)
- `--output-dir, -o`: Output directory for results
- `--results-csv`: Output CSV filename (default: `phenology_results.csv`)
- `--visualization-png`: Output visualization filename (default: `phenology_analysis.png`)

### 3. Check Version

Display the application version:

```bash
python -m phenocover --version
```

## Output Files

The analysis generates the following files (by default in `results/` directory):

1. **results/phenology_results.csv**: Daily predictions including:
   - Date
   - NDVI values (observed and interpolated)
   - Fractional Vegetation Cover (FVC)
   - Ground cover percentage
   - Growth stages
   - Weather data (temperature, GDD, precipitation)
   - Stress indices

2. **results/phenology_analysis.png**: Comprehensive visualization with multiple subplots:
   - NDVI and FVC time series
   - Growth stages timeline
   - Weather data integration
   - Agricultural stress indices

3. **Logs**: Stored in `./logs/` directory
   - Application logs with timestamps
   - Error tracking and debugging information

**Note:** Output files are created in `results/` directory by default. You can customize the output location and filenames using config files or CLI parameters.

## Examples

### Example 1: Quick Start with Sample Data

```bash
# Generate configuration file
python -m phenocover generate-config --format yaml --output my_config.yml

# Edit my_config.yml with your data paths
# Then run analysis
python -m phenocover phenology-analyzer --config my_config.yml
```

### Example 2: Custom Analysis Parameters

```bash
python -m phenocover phenology-analyzer \
  --ndvi-file "path/to/ndvi_data.csv" \
  --sowing-date "15.09.2023" \
  --harvest-date "20.08.2024" \
  --geojson-file "path/to/field.geojson" \
  --fvc-method dimyuk \
  --interpolation-method cubic \
  --output-dir "./analysis_results"
```

### Example 3: Custom Output Filenames

```bash
# Specify custom output filenames
python -m phenocover phenology-analyzer \
  --config config.yml \
  --results-csv "field_A_results.csv" \
  --visualization-png "field_A_analysis.png"

# Use in batch processing for multiple fields
python -m phenocover phenology-analyzer \
  --config field1_config.yml \
  --results-csv "field1_results.csv" \
  --visualization-png "field1_viz.png"
```

### Example 4: Testing Different Methods

```bash
# Test with seasonal FVC method
python -m phenocover phenology-analyzer --config config.yml --fvc-method seasonal

# Test with dimyuk FVC method
python -m phenocover phenology-analyzer --config config.yml --fvc-method dimyuk

# Test with custom interpolation
python -m phenocover phenology-analyzer --config config.yml --interpolation-method cubic
```

## Configuration File Details

### FVC Methods

- **seasonal**: Adaptive seasonal-based estimation (recommended for most cases)
- **dimyuk**: Based on Dimyuk et al. formula
- **custom**: User-defined parameters

### Interpolation Methods

- **balanced**: Combines linear and cubic interpolation (recommended)
- **linear**: Simple linear interpolation
- **cubic**: Cubic spline interpolation

### Analysis Options

Control which analysis steps to perform:

```yaml
analysis_options:
  estimate_fvc: true          # Estimate Fractional Vegetation Cover
  interpolate_ndvi: true      # Interpolate NDVI values
  estimate_growth_stages: true # Estimate wheat growth stages
  create_visualization: true   # Create plots
  save_results: true          # Save CSV results
  generate_summary: true      # Generate text summary
```

## Logging

The CLI uses rich console output and structured logging:

- **Console Output**: Formatted with Rich library for better readability
- **Progress Indicators**: Real-time progress for long-running operations
- **Log Files**: Detailed logs saved to `./logs/` directory
- **Log Levels**: INFO for standard output, DEBUG for detailed diagnostics

## Troubleshooting

### Missing Required Parameters

If you see an error about missing parameters:

```text
Error: Missing required parameters: ndvi_file, sowing_date
```

Solution: Provide all required parameters via config file or CLI options.

### Invalid Date Format

Dates must be in DD.MM.YYYY format:

- ✅ Correct: `"03.10.2023"`
- ❌ Incorrect: `"2023-10-03"` or `"10/03/2023"`

### File Not Found

Ensure all file paths are correct:

- Use absolute paths or paths relative to the current working directory
- Check file extensions match exactly
- Verify GeoJSON file is valid JSON format

### Configuration File Errors

If your YAML/JSON file has syntax errors:

- Use a YAML/JSON validator
- Check for proper indentation (YAML is indent-sensitive)
- Ensure quotes are balanced

## Support

For issues, questions, or contributions:

- GitHub: <https://github.com/joemureithi/phenocover>

## License

MIT License - See LICENSE file for details
