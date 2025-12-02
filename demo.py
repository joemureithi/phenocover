#!/usr/bin/env python3
"""
Demo script for Weather-Enhanced Wheat Phenology Analysis
"""

from phenocover.wheat_phenology_analyzer import WheatPhenologyAnalyzer
from phenocover.logging import configure_logging, get_logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

__author__ = "Muhammad Arslan"
__email__ = "arslanhoney1599@gmail.com"
__year__ = "2025"

# Configure logging
configure_logging(
    level="INFO",
    log_dir="./logs",
    enable_file_logging=True,
    enable_console_logging=True,
    use_rich=True,
    suppress_third_party_debug=True
)

logger = get_logger(__name__)
console = Console()


def main():
    """Run the wheat phenology analysis with weather integration"""

    # Display header
    console.print(Panel.fit(
        "[bold cyan]WEATHER-ENHANCED WHEAT PHENOLOGY ANALYSIS[/bold cyan]\n\n"
        "[green]Real weather data from Open-Meteo API (free, no API key)[/green]\n"
        "• Weather-informed growth stage estimation\n"
        "• Ground cover percentage calculation\n"
        "• Agricultural stress indices\n",
        border_style="cyan"
    ))

    logger.info("Starting wheat phenology analysis demo")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Initialize analyzer with weather integration
        task = progress.add_task(
            "[cyan]Initializing analyzer with real weather data...", total=None)
        logger.info("Initializing WheatPhenologyAnalyzer with parameters")
        analyzer = WheatPhenologyAnalyzer(
            ndvi_file='NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv',
            sowing_date='03.10.2023',
            harvest_date='30.07.2024',
            geojson_file='field_location.geojson'
        )
        progress.update(task, completed=True)
        console.print(
            "[green]✓[/green] Analyzer initialized with weather integration")

        # Run complete analysis
        task = progress.add_task(
            "[cyan]Estimating FVC parameters...", total=None)
        logger.info("Estimating FVC parameters using seasonal method")
        analyzer.estimate_fvc_parameters(method='seasonal')
        progress.update(task, completed=True)
        console.print("[green]✓[/green] FVC parameters estimated")

        task = progress.add_task(
            "[cyan]Interpolating NDVI data with weather integration...", total=None)
        logger.info("Interpolating NDVI data using balanced method")
        analyzer.interpolate_ndvi(method='balanced')
        progress.update(task, completed=True)
        console.print("[green]✓[/green] NDVI interpolation complete")

        task = progress.add_task(
            "[cyan]Estimating growth stages with weather data...", total=None)
        logger.info("Estimating growth stages with weather integration")
        analyzer.estimate_growth_stages()
        progress.update(task, completed=True)
        console.print("[green]✓[/green] Growth stages estimated")

        task = progress.add_task(
            "[cyan]Creating enhanced visualization...", total=None)
        logger.info("Creating visualization plots")
        analyzer.create_visualization()
        progress.update(task, completed=True)
        console.print("[green]✓[/green] Visualization created")

        task = progress.add_task("[cyan]Saving results...", total=None)
        logger.info("Saving results to CSV file")
        analyzer.save_results()
        progress.update(task, completed=True)
        console.print("[green]✓[/green] Results saved")

        task = progress.add_task(
            "[cyan]Generating summary report...", total=None)
        logger.info("Generating summary report")
        analyzer.generate_summary_report()
        progress.update(task, completed=True)
        console.print("[green]✓[/green] Summary report generated")

    # Display completion message
    console.print(Panel.fit(
        "[bold green]ANALYSIS COMPLETE![/bold green]\n\n"
        "[bold]Generated files:[/bold]\n"
        "• [cyan]wheat_phenology_results.csv[/cyan]: Daily predictions with weather data\n"
        "• [cyan]wheat_phenology_analysis.png[/cyan]: Enhanced visualization\n\n"
        "[bold]Key Features:[/bold]\n"
        "• Real weather data from Open-Meteo API\n"
        "• Growing Degree Days (GDD) calculation\n"
        "• Weather-informed growth stage estimation\n"
        "• Ground cover percentage estimation\n"
        "• Agricultural stress indices\n"
        "• Location-specific analysis for Bavaria, Germany",
        border_style="green"
    ))

    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()
