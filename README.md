# PhenoCover - Weather-Enhanced Wheat Phenology Analysis

A comprehensive scientific tool for analyzing wheat phenology and estimating ground cover using NDVI observations integrated with real weather data from Open-Meteo API.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Real Weather Data Integration**: Uses Open-Meteo API (completely free, no API key required)
- **Weather-Informed Growth Stages**: Estimates growth stages using Growing Degree Days (GDD)
- **Ground Cover Estimation**: Calculates Fractional Vegetation Cover (FVC) and ground cover percentage
- **Agricultural Stress Indices**: Heat stress, cold stress, drought stress calculations
- **Enhanced Visualizations**: Multi-panel plots with weather data integration
- **Location-Specific Analysis**: Tailored for specific field locations
- **Uncertainty Quantification**: Realistic confidence intervals for predictions
- **CLI and Python API**: Use via command-line or integrate into your Python projects

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/tum-gis/phenocover.git
cd phenocover

# Install the package
pip install .

# Or install in development mode
pip install -e .
```

For detailed installation instructions, see [INSTALL.md](/phenocover/docs/INSTALL.md).

## Quick Start

### 1. Verify Installation

```bash
phenocover --version
```

### 2. Run Analysis

#### Option A: Using Demo Script

```bash
python demo.py
```

#### Option B: Using CLI (Recommended)

```bash
# Generate a sample configuration file
python -m phenocover generate-config --format yaml --output config.yml

# Edit config.yml with your data paths, then run:
python -m phenocover phenology-analyzer --config config.yml

# Or use command-line parameters directly:
python -m phenocover phenology-analyzer \
  --ndvi-file "NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv" \
  --sowing-date "03.10.2023" \
  --harvest-date "30.07.2024" \
  --geojson-file "field_location.geojson"
```

See [CLI_USAGE.md](/phenocover/docs/CLI_USAGE.md) for comprehensive CLI documentation.

### 3. Custom Analysis (Python API)

```python
from phenocover.wheat_phenology_analyzer import WheatPhenologyAnalyzer

# Initialize analyzer
analyzer = WheatPhenologyAnalyzer(
    ndvi_file='NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv',
    sowing_date='03.10.2023',
    harvest_date='30.07.2024',
    geojson_file='field_location.geojson'
)

# Run analysis
analyzer.estimate_fvc_parameters(method='seasonal')
analyzer.interpolate_ndvi(method='balanced')
analyzer.estimate_growth_stages()
analyzer.create_visualization()
analyzer.save_results()
analyzer.generate_summary_report()
```

## Repository Structure

```text
wheat_phenology_weather_enhanced/
├── wheat_phenology_analyzer.py          # Main analysis engine
├── demo.py                              # Demonstration script
├── setup.py                             # Installation script
├── requirements.txt                     # Python dependencies
├── field_location.geojson              # Field coordinates (Bavaria, Germany)
├── NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv  # NDVI observations
├── README.md                           # This documentation
├── wheat_phenology_results.csv         # Output: Daily predictions
└── wheat_phenology_analysis.png         # Output: Enhanced visualization
```

### Key Files

- **`wheat_phenology_analyzer.py`**: Core analysis engine with weather integration, NDVI interpolation, and growth stage estimation
- **`field_location.geojson`**: GeoJSON coordinates for Bavaria, Germany (49.2076°N, 10.6276°E)
- **`NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv`**: Sparse NDVI observations (5 data points)
- **Output files**: Generated daily predictions and multi-panel visualizations

## Pipeline Workflow

### Overview

The analysis pipeline transforms sparse NDVI observations into comprehensive daily phenology data through seven sequential phases:

```text
Input Data → Weather Integration → NDVI Processing → Growth Analysis → Visualization → Output
```

### Workflow Phases

#### 1. Data Initialization

- Load GeoJSON coordinates for weather data fetching
- Fetch historical weather data from Open-Meteo API
- Load and validate sparse NDVI observations
- Calculate Growing Degree Days (GDD) with base temperature 0°C

#### 2. FVC Parameter Estimation

- **Literature-based**: NDVI_soil = 0.15, NDVI_vegetation = 0.85
- **Data-driven**: Based on observed NDVI min/max values
- **Seasonal**: Using early and mid-season patterns

#### 3. NDVI Interpolation

- Apply cubic spline interpolation for smooth daily time series
- Calculate realistic confidence intervals (15% uncertainty factor)
- Ensure physiologically valid NDVI range [0,1]
- Compute FVC and ground cover percentages

#### 4. Growth Stage Estimation

- Calculate cumulative GDD from sowing date
- Apply wheat-specific GDD thresholds:
  - Sowing (0), Emergence (50), Tillering (200), Stem Elongation (500)
  - Booting (800), Heading (1000), Flowering (1200), Grain Filling (1400)
  - Maturity (1800), Harvest (2000)
- Identify exact transition dates and handle overlapping stages

#### 5. Stress Analysis

- **Heat Stress**: Maximum temperature > 30°C
- **Cold Stress**: Minimum temperature < -5°C
- **Drought Stress**: Consecutive days without precipitation
- Correlate stress periods with growth stages

#### 6. Visualization

- Generate multi-panel plots with growth stage markers
- Include NDVI trends, ground cover, weather conditions, and GDD
- Add vertical lines and labels for each growth stage transition
- Apply professional formatting and high-resolution output

#### 7. Results Compilation

- Combine all parameters into comprehensive daily dataset
- Generate summary statistics by growth stage
- Create detailed CSV output and visualization files

## Technical Approach

### Core Concept

The pipeline transforms sparse NDVI observations (5-20 data points) into comprehensive daily phenology data through weather-enhanced analysis and robust uncertainty quantification.

### Key Features

#### Weather-Enhanced Phenology

- Real weather data integration via Open-Meteo API
- Growing Degree Days calculation with actual temperature data
- Location-specific analysis and stress assessment
- Weather-informed growth stage timing

#### Uncertainty Quantification

- Physiologically constrained confidence intervals
- Realistic uncertainty estimates based on observed data variability
- Avoids unrealistic extrapolation beyond valid NDVI ranges [0,1]
- Handles sparse data scenarios appropriately

#### Multi-Method Flexibility

- Multiple FVC parameter estimation methods
- Configurable growth stage thresholds
- Adaptable interpolation approaches
- Comprehensive stress index calculations

### Data Processing Flow

1. **Input**: GeoJSON coordinates → Weather API → NDVI CSV
2. **Processing**: GDD calculation → NDVI interpolation → Growth stages
3. **Output**: Daily predictions → Visualizations → Summary reports

### Error Handling

- API fallback mechanisms for connection failures
- Input validation and format checking
- Robust interpolation for edge cases
- Quality control for output data

## Weather Data Integration

### Data Source

- **API**: Open-Meteo Archive API
- **URL**: <https://archive-api.open-meteo.com/v1/archive>
- **Cost**: Completely free
- **Registration**: Not required
- **API Key**: Not required

### Weather Parameters

- Temperature (mean, min, max)
- Precipitation
- Relative humidity
- Atmospheric pressure
- Wind speed
- Cloud cover
- Growing Degree Days (GDD)
- Agricultural stress indices

### Location

- **Field**: Bavaria, Germany
- **Coordinates**: 49.2076°N, 10.6276°E
- **Analysis Period**: October 3, 2023 to July 30, 2024

## Analysis Components

### 1. NDVI Interpolation

- **Balanced Interpolation**: Combines physiological knowledge with smooth transitions
- **Confidence Intervals**: Realistic uncertainty quantification based on data variability
- **Physiological Constraints**: Ensures NDVI values stay within valid range [0, 1]

### 2. FVC Estimation

- **Fractional Vegetation Cover**: Calculated from NDVI using linear mixing model
- **Ground Cover Percentage**: FVC expressed as percentage
- **Parameter Estimation**: Literature-based, data-driven, or seasonal methods

### 3. Growth Stage Estimation

- **Weather-Informed**: Uses Growing Degree Days (GDD) thresholds
- **Stages**: Sowing, Emergence, Tillering, Stem Elongation, Booting, Heading, Flowering, Grain Filling, Maturity, Harvest
- **GDD Thresholds**: Each stage has specific GDD requirements

### 4. Agricultural Stress Indices

- **Heat Stress**: Temperature > 30°C
- **Cold Stress**: Temperature < -5°C
- **Drought Stress**: Consecutive days without precipitation
- **Optimal Conditions**: Temperature 10-25°C with precipitation

## Output Files

### 1. Results CSV (`phenology_results.csv`)

Daily predictions including:

- NDVI interpolated values with confidence intervals
- FVC and ground cover percentage
- Growth stage assignments
- Weather data (temperature, precipitation, humidity, etc.)
- Growing Degree Days
- Agricultural stress indices

### 2. Visualization (`phenology_analysis.png`)

Multi-panel plot showing:

- NDVI time series with confidence intervals and growth stage markers
- Ground cover percentage with uncertainty bounds
- Weather conditions (temperature and precipitation)
- Growing Degree Days with stage threshold lines

#### Visualization Features

The generated visualization includes:

- **Panel 1 (Top Left)**: NDVI time series with confidence intervals (shaded area) and vertical growth stage markers
- **Panel 2 (Top Right)**: Ground cover percentage showing vegetation coverage over time
- **Panel 3 (Bottom Left)**: Weather conditions with temperature (line) and precipitation (bars)
- **Panel 4 (Bottom Right)**: Cumulative Growing Degree Days with horizontal threshold lines for each growth stage

Each growth stage transition is marked with:

- Vertical dashed lines indicating exact transition dates
- Color-coded labels showing stage names
- Precise timing for each growth phase

The visualization provides a comprehensive view of wheat development throughout the growing season, integrating NDVI observations, weather conditions, and phenological stages.

#### Sample Output Visualization

The pipeline generates a high-resolution multi-panel visualization showing the complete wheat phenology analysis:

![Wheat Phenology Analysis](wheat_phenology_analysis.png)

*Note: The actual visualization file (`wheat_phenology_analysis.png`) is generated when running the analysis and contains the complete multi-panel plot with growth stage markers, confidence intervals, and weather data integration.*

## Configuration

### Growth Stage GDD Thresholds

```python
growth_stages = {
    'Sowing': {'gdd_threshold': 0},
    'Emergence': {'gdd_threshold': 50},
    'Tillering': {'gdd_threshold': 200},
    'Stem Elongation': {'gdd_threshold': 500},
    'Booting': {'gdd_threshold': 800},
    'Heading': {'gdd_threshold': 1000},
    'Flowering': {'gdd_threshold': 1200},
    'Grain Filling': {'gdd_threshold': 1400},
    'Maturity': {'gdd_threshold': 1800},
    'Harvest': {'gdd_threshold': 2000}
}
```

### FVC Parameter Methods

- **Literature**: NDVI_soil = 0.15, NDVI_vegetation = 0.85
- **Data-driven**: Based on observed NDVI min/max values
- **Seasonal**: Based on early and mid-season NDVI patterns

## Key Benefits

### Weather Integration

- **Real Data**: Actual weather data, not synthetic
- **Location-Specific**: Tailored to field location
- **Weather-Informed**: Growth stages based on accumulated heat units
- **Stress Awareness**: Identifies weather-related stress conditions

### Scientific Accuracy

- **Physiological Knowledge**: Incorporates wheat growth patterns
- **Uncertainty Quantification**: Realistic confidence intervals for predictions
- **Multiple Methods**: Various interpolation and estimation approaches
- **Robust Estimation**: Designed for sparse data scenarios

### Practical Applications

- **Crop Management**: Timing of agricultural operations
- **Yield Prediction**: Growth stage-based yield estimation
- **Stress Monitoring**: Early detection of weather stress
- **Research**: Phenology studies and climate impact assessment

## Scientific Background

### NDVI (Normalized Difference Vegetation Index)

- **Formula**: NDVI = (NIR - Red) / (NIR + Red)
- **Range**: -1 to +1 (typically 0 to 1 for vegetation)
- **Interpretation**: Higher values indicate more vegetation

### FVC (Fractional Vegetation Cover)

- **Formula**: FVC = (NDVI - NDVI_soil) / (NDVI_vegetation - NDVI_soil)
- **Range**: 0 to 1
- **Interpretation**: Proportion of ground covered by vegetation

### Growing Degree Days (GDD)

- **Formula**: GDD = max(0, (T_max + T_min) / 2 - T_base)
- **Base Temperature**: 0°C for wheat
- **Interpretation**: Accumulated heat units for plant development

## Installation

### Option 1: Manual Installation

```bash
pip install -r requirements.txt
```

### Option 2: Automated Setup

```bash
python setup.py
```

## Usage Examples

### Basic Analysis

```python
from wheat_phenology_analyzer import WheatPhenologyAnalyzer

analyzer = WheatPhenologyAnalyzer(
    ndvi_file='your_ndvi_data.csv',
    sowing_date='03.10.2023',
    harvest_date='30.07.2024',
    geojson_file='field_location.geojson'
)

analyzer.estimate_fvc_parameters(method='seasonal')
analyzer.interpolate_ndvi(method='balanced')
analyzer.estimate_growth_stages()
analyzer.create_visualization()
analyzer.save_results()
```

### Custom Configuration

```python
# Custom FVC parameters
analyzer.estimate_fvc_parameters(method='literature')

# Different interpolation method
analyzer.interpolate_ndvi(method='linear')

# Custom output filename
analyzer.save_results('custom_results.csv')
analyzer.create_visualization('custom_plot.png')
```

## Data Requirements

### NDVI Data Format

CSV file with columns:

- `phenomenonTime`: Date/time of observation (YYYY-MM-DD HH:MM:SS)
- `NDVI`: NDVI values (0-1 range)

### Location Data Format

GeoJSON file with polygon coordinates for the field location.

### Date Format

- Sowing date: DD.MM.YYYY (e.g., '03.10.2023')
- Harvest date: DD.MM.YYYY (e.g., '30.07.2024')

## Troubleshooting

### Common Issues

1. **API Connection Errors**: The tool automatically falls back to synthetic weather data if the API is unavailable.

2. **Missing Files**: Ensure all required files are present in the working directory.

3. **Date Format Errors**: Use the correct date format (DD.MM.YYYY) for sowing and harvest dates.

4. **NDVI Data Issues**: Verify NDVI values are within the valid range [0, 1].

### Performance Notes

- The tool is optimized for sparse NDVI data (5-20 observations)
- Weather data fetching may take 10-30 seconds depending on internet connection
- Analysis typically completes in under 1 minute for a full growing season

## Contributing

This tool is designed for agricultural research and crop management applications. Contributions are welcome for:

- Additional weather data sources
- Enhanced interpolation methods
- New agricultural stress indices
- Improved visualization options
- Additional crop types beyond wheat

## Acknowledgments

- Open-Meteo API for providing free weather data
- Agricultural research community for phenology models
- Remote sensing community for NDVI applications

---

For questions or support, please refer to the code documentation or create an issue in the repository.
