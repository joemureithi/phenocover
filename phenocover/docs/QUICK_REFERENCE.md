# PhenoCover CLI Quick Reference

## Installation

```bash
pip install -r requirements.txt
```

## Commands

### Generate Config File

```bash
# YAML (recommended)
python -m phenocover generate-config --format yaml --output config.yml

# JSON
python -m phenocover generate-config --format json --output config.json
```

### Run Analysis

#### With Config File

```bash
python -m phenocover phenology-analyzer --config config.yml
```

#### With CLI Parameters

```bash
python -m phenocover phenology-analyzer \
  --ndvi-file "data.csv" \
  --sowing-date "03.10.2023" \
  --harvest-date "30.07.2024" \
  --geojson-file "field.geojson"
```

#### Hybrid (Config + Override)

```bash
python -m phenocover phenology-analyzer \
  --config config.yml \
  --fvc-method custom \
  --output-dir ./results
```

### Check Version

```bash
python -m phenocover --version
```

### Help

```bash
python -m phenocover --help
python -m phenocover phenology-analyzer --help
python -m phenocover generate-config --help
```

## Common Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Config file path | None |
| `--ndvi-file` | `-n` | NDVI CSV file | Required |
| `--sowing-date` | `-s` | Sowing date (DD.MM.YYYY) | Required |
| `--harvest-date` | `-h` | Harvest date (DD.MM.YYYY) | Required |
| `--geojson-file` | `-g` | GeoJSON location file | Required |
| `--fvc-method` | - | FVC method (seasonal/dimyuk/custom) | seasonal |
| `--interpolation-method` | - | Interpolation (balanced/linear/cubic) | balanced |
| `--output-dir` | `-o` | Output directory | Current dir |
| `--results-csv` | - | Output CSV filename | phenology_results.csv |
| `--visualization-png` | - | Output PNG filename | phenology_analysis.png |

## Configuration File Example

```yaml
ndvi_file: "data.csv"
sowing_date: "03.10.2023"
harvest_date: "30.07.2024"
geojson_file: "field.geojson"
fvc_method: "seasonal"
interpolation_method: "balanced"
output_dir: "./results"
results_csv: "results/phenology_results.csv"
visualization_png: "results/phenology_analysis.png"
```

## Output Files

- `results/phenology_results.csv` (or custom name) - Daily predictions with weather
- `results/phenology_analysis.png` (or custom name) - Visualization plots
- `logs/` - Application logs

**Note:** Files are created in `results/` directory by default.

## Typical Workflow

1. **Generate config**: `python -m phenocover generate-config`
2. **Edit config**: Update file paths and parameters
3. **Run analysis**: `python -m phenocover phenology-analyzer --config config.yml`
4. **Check results**: Review CSV and PNG outputs

## Tips

- Use YAML for configs (more readable)
- CLI options override config file
- Check logs/ directory for debugging
- Date format must be DD.MM.YYYY
- Use absolute paths or relative to working directory

## Documentation

- Full guide: [CLI_USAGE](CMD_USAGE.md)
- Implementation: `IMPLEMENTATION_SUMMARY.md`
