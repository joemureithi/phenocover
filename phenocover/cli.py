#!/usr/bin/env python3
"""
Command Line Interface for PhenoCover

This module provides the command-line interface for the PhenoCover application,
which enables weather-enhanced wheat phenology analysis and ground cover estimation.

The CLI supports various operations including:
- Running phenology analysis with configurable parameters
- Generating sample configuration files (YAML/JSON)
- Weather data integration and analysis
- Vegetation cover estimation

Author: Joseph Gitahi
Created: 2025
License: MIT License
Repository: https://github.com/joemureithi/phenocover

Key Dependencies:
    - typer: Modern CLI framework
    - rich: Rich text and beautiful formatting
    - pyyaml: YAML configuration support

Usage:
    python -m phenocover --help
    python -m phenocover phenology-analyzer --config config.yml
    python -m phenocover phenology-analyzer --ndvi-file data.csv --sowing-date "03.10.2023"
    python -m phenocover generate-config --format yaml --output config.yml
"""

import typer
import json
import yaml
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from phenocover import __app_name__, __version__
from phenocover.logging import configure_logging, get_logger
from phenocover.utils import clear
from phenocover.wheat_phenology_analyzer import WheatPhenologyAnalyzer

# Logger
logger = get_logger(__name__)
configure_logging(
    level="INFO",
    log_dir="./logs",
    enable_file_logging=True,
    enable_console_logging=True,
    use_rich=True,
    suppress_third_party_debug=True
)
# Main app
app = typer.Typer()
console = Console()


# Main callback


def _version_callback(value: bool) -> None:
    clear()
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    """
    PhenoCover - Weather-Enhanced Wheat Phenology Analysis Tool
    """
    return


@app.command(name="phenology-analyzer")
def phenology_analyzer(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (YAML or JSON)",
        exists=True,
        dir_okay=False,
    ),
    ndvi_file: Optional[str] = typer.Option(
        None,
        "--ndvi-file",
        "-n",
        help="Path to NDVI CSV file",
    ),
    sowing_date: Optional[str] = typer.Option(
        None,
        "--sowing-date",
        "-s",
        help="Sowing date (format: DD.MM.YYYY)",
    ),
    harvest_date: Optional[str] = typer.Option(
        None,
        "--harvest-date",
        "-h",
        help="Harvest date (format: DD.MM.YYYY)",
    ),
    geojson_file: Optional[str] = typer.Option(
        None,
        "--geojson-file",
        "-g",
        help="Path to GeoJSON file with field location",
    ),
    fvc_method: str = typer.Option(
        "seasonal",
        "--fvc-method",
        help="FVC estimation method (seasonal, dimyuk, or custom)",
    ),
    interpolation_method: str = typer.Option(
        "balanced",
        "--interpolation-method",
        help="NDVI interpolation method (balanced, linear, or cubic)",
    ),
    # output_dir: Optional[str] = typer.Option(
    #     None,
    #     "--output-dir",
    #     "-o",
    #     help="Output directory for results",
    # ),
    results_csv: Optional[str] = typer.Option(
        None,
        "--results-csv",
        help="Output CSV filename (default: phenology_results.csv)",
    ),
    visualization_png: Optional[str] = typer.Option(
        None,
        "--visualization-png",
        help="Output visualization filename (default: phenology_analysis.png)",
    ),
) -> None:
    """
    Run weather-enhanced wheat phenology analysis.

    Parameters can be provided via command-line options or a configuration file.
    Command-line options override configuration file values.
    """
    try:
        # Load configuration from file if provided
        params = {}
        if config:
            logger.info(f"Loading configuration from: {config}")
            params = _load_config(config)
            console.print(
                f"[green]✓[/green] Configuration loaded from {config}")

        # Override with command-line arguments
        if ndvi_file:
            params['ndvi_file'] = ndvi_file
        if sowing_date:
            params['sowing_date'] = sowing_date
        if harvest_date:
            params['harvest_date'] = harvest_date
        if geojson_file:
            params['geojson_file'] = geojson_file
        # if output_dir:
        #     params['output_dir'] = output_dir
        if results_csv:
            params['results_csv'] = results_csv
        if visualization_png:
            params['visualization_png'] = visualization_png

        params['fvc_method'] = fvc_method
        params['interpolation_method'] = interpolation_method

        # Set defaults for output filenames if not provided
        # Create results directory if it doesn't exist
        # results_dir = Path('results')
        # results_dir.mkdir(exist_ok=True)

        if 'results_csv' not in params or params['results_csv'] is None:
            params['results_csv'] = str('phenology_results.csv')
        elif not Path(params['results_csv']).is_absolute() and '/' not in params['results_csv'] and '\\' not in params['results_csv']:
            # If it's just a filename (no path), put it in results
            params['results_csv'] = str(params['results_csv'])

        if 'visualization_png' not in params or params['visualization_png'] is None:
            params['visualization_png'] = str(
                'phenology_analysis.png')
        elif not Path(params['visualization_png']).is_absolute() and '/' not in params['visualization_png'] and '\\' not in params['visualization_png']:
            # If it's just a filename (no path), put it in results
            params['visualization_png'] = str(
                params['visualization_png'])
        # Validate required parameters
        required_params = ['ndvi_file', 'sowing_date',
                           'harvest_date', 'geojson_file']
        missing_params = [
            p for p in required_params if p not in params or params[p] is None]

        if missing_params:
            console.print(
                f"[red]Error:[/red] Missing required parameters: {', '.join(missing_params)}")
            console.print(
                "\nProvide them via --config file or command-line options.")
            console.print(
                "Use 'phenocover generate-config' to create a sample configuration file.")
            raise typer.Exit(1)

        # Display analysis header
        _display_header()

        # Display configuration
        _display_config(params)

        # Run analysis
        _run_analysis(params)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        console.print(f"[red]✗[/red] Analysis failed: {str(e)}")
        raise typer.Exit(1)


@app.command(name="generate-config")
def generate_config(
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Configuration format (yaml or json)",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: config.yaml or config.json)",
    ),
) -> None:
    """
    Generate a sample configuration file for phenology analysis.

    Creates a template configuration file with all available parameters
    and example values.
    """
    try:
        # Sample configuration
        sample_config = {
            "ndvi_file": "NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv",
            "sowing_date": "03.10.2023",
            "harvest_date": "30.07.2024",
            "geojson_file": "field_location.geojson",
            "fvc_method": "seasonal",
            "interpolation_method": "balanced",
            "output_dir": "./results",
            "results_csv": "results/phenology_results.csv",
            "visualization_png": "results/phenology_analysis.png",
            "analysis_options": {
                "estimate_fvc": True,
                "interpolate_ndvi": True,
                "estimate_growth_stages": True,
                "create_visualization": True,
                "save_results": True,
                "generate_summary": True
            },
            "weather_integration": {
                "enabled": True,
                "source": "Open-Meteo API"
            }
        }

        # Determine output file
        if output is None:
            output = Path(f"config.{format}")

        # Write configuration
        with open(output, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(sample_config, f,
                          default_flow_style=False, sort_keys=False)
            elif format.lower() == 'json':
                json.dump(sample_config, f, indent=2)
            else:
                console.print(
                    f"[red]Error:[/red] Unsupported format '{format}'. Use 'yaml' or 'json'.")
                raise typer.Exit(1)

        console.print(
            f"[green]✓[/green] Sample configuration file created: {output}")
        console.print("\nEdit the file and run:")
        console.print(f"  phenocover phenology-analyzer --config {output}")

        logger.info(f"Generated sample configuration file: {output}")

    except Exception as e:
        logger.error(f"Failed to generate config: {str(e)}", exc_info=True)
        console.print(
            f"[red]✗[/red] Failed to generate configuration: {str(e)}")
        raise typer.Exit(1)


def _load_config(config_path: Path) -> dict:
    """Load configuration from YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: {config_path.suffix}")


def _display_header():
    """Display analysis header."""
    console.print(Panel.fit(
        "[bold cyan]WEATHER-ENHANCED WHEAT PHENOLOGY ANALYSIS[/bold cyan]\n"
        "Real weather data from Open-Meteo API (free, no API key)\n"
        "Weather-informed growth stage estimation\n"
        "Ground cover percentage calculation\n"
        "Agricultural stress indices",
        border_style="cyan"
    ))


def _display_config(params: dict):
    """Display configuration table."""
    table = Table(title="Analysis Configuration",
                  show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("NDVI File", params.get('ndvi_file', 'N/A'))
    table.add_row("Sowing Date", params.get('sowing_date', 'N/A'))
    table.add_row("Harvest Date", params.get('harvest_date', 'N/A'))
    table.add_row("GeoJSON File", params.get('geojson_file', 'N/A'))
    table.add_row("FVC Method", params.get('fvc_method', 'seasonal'))
    table.add_row("Interpolation Method", params.get(
        'interpolation_method', 'balanced'))
    if params.get('output_dir'):
        table.add_row("Output Directory", params['output_dir'])
    table.add_row("Results CSV", params.get(
        'results_csv', 'phenology_results.csv'))
    table.add_row("Visualization PNG", params.get(
        'visualization_png', 'phenology_analysis.png'))

    console.print(table)


def _run_analysis(params: dict):
    """Run the phenology analysis."""
    logger.info("Starting phenology analysis")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Initialize analyzer
        task = progress.add_task(
            "[cyan]Initializing analyzer with real weather data...", total=None)
        logger.info("Initializing WheatPhenologyAnalyzer")
        analyzer = WheatPhenologyAnalyzer(
            ndvi_file=params['ndvi_file'],
            sowing_date=params['sowing_date'],
            harvest_date=params['harvest_date'],
            geojson_file=params['geojson_file']
        )
        progress.update(task, completed=True)
        console.print(
            "[green]✓[/green] Analyzer initialized with weather integration")

        # Estimate FVC parameters
        task = progress.add_task(
            "[cyan]Estimating FVC parameters...", total=None)
        logger.info(
            f"Estimating FVC parameters using method: {params['fvc_method']}")
        analyzer.estimate_fvc_parameters(method=params['fvc_method'])
        progress.update(task, completed=True)
        console.print("[green]✓[/green] FVC parameters estimated")

        # Interpolate NDVI
        task = progress.add_task(
            "[cyan]Interpolating NDVI data with weather integration...", total=None)
        logger.info(
            f"Interpolating NDVI using method: {params['interpolation_method']}")
        analyzer.interpolate_ndvi(method=params['interpolation_method'])
        progress.update(task, completed=True)
        console.print("[green]✓[/green] NDVI interpolation complete")

        # Estimate growth stages
        task = progress.add_task(
            "[cyan]Estimating growth stages with weather data...", total=None)
        logger.info("Estimating growth stages")
        analyzer.estimate_growth_stages()
        progress.update(task, completed=True)
        console.print("[green]✓[/green] Growth stages estimated")

        # Create visualization
        task = progress.add_task(
            "[cyan]Creating enhanced visualization...", total=None)
        logger.info(f"Creating visualization: {params['visualization_png']}")
        analyzer.create_visualization(save_path=params['visualization_png'])
        progress.update(task, completed=True)
        console.print(
            f"[green]✓[/green] Visualization saved to {params['visualization_png']}")

        # Save results
        task = progress.add_task("[cyan]Saving results...", total=None)
        logger.info(f"Saving results to: {params['results_csv']}")
        analyzer.save_results(output_file=params['results_csv'])
        progress.update(task, completed=True)
        console.print(
            f"[green]✓[/green] Results saved to {params['results_csv']}")

        # Generate summary
        task = progress.add_task(
            "[cyan]Generating summary report...", total=None)
        logger.info("Generating summary report")
        analyzer.generate_summary_report()
        progress.update(task, completed=True)
        console.print("[green]✓[/green] Summary report generated")

    # Display completion message
    console.print(Panel.fit(
        f"[bold green]ANALYSIS COMPLETE![/bold green]\n\n"
        f"[bold]Generated files:[/bold]\n"
        f"• {params['results_csv']}: Daily predictions with weather data\n"
        f"• {params['visualization_png']}: Enhanced visualization\n\n"
        f"[bold]Key Features:[/bold]\n"
        f"• Real weather data from Open-Meteo API\n"
        f"• Growing Degree Days (GDD) calculation\n"
        f"• Weather-informed growth stage estimation\n"
        f"• Ground cover percentage estimation\n"
        f"• Agricultural stress indices\n"
        f"• Location-specific analysis",
        border_style="green"
    ))

    logger.info("Analysis completed successfully")
