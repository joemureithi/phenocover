#!/usr/bin/env python3
"""
Demo script for Weather-Enhanced Wheat Phenology Analysis
"""

from wheat_phenology_analyzer import WheatPhenologyAnalyzer

def main():
    """Run the wheat phenology analysis with weather integration"""
    print("="*70)
    print("WEATHER-ENHANCED WHEAT PHENOLOGY ANALYSIS")
    print("="*70)
    print("Real weather data from Open-Meteo API (free, no API key)")
    print("Weather-informed growth stage estimation")
    print("Ground cover percentage calculation")
    print("Agricultural stress indices")
    print()
    
    # Initialize analyzer with weather integration
    print("Initializing analyzer with real weather data...")
    analyzer = WheatPhenologyAnalyzer(
        ndvi_file='NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv',
        sowing_date='03.10.2023',
        harvest_date='30.07.2024',
        geojson_file='field_location.geojson'
    )
    
    # Run complete analysis
    print("\nEstimating FVC parameters...")
    analyzer.estimate_fvc_parameters(method='seasonal')
    
    print("Interpolating NDVI data with weather integration...")
    analyzer.interpolate_ndvi(method='balanced')
    
    print("Estimating growth stages with weather data...")
    analyzer.estimate_growth_stages()
    
    print("Creating enhanced visualization...")
    analyzer.create_visualization()
    
    print("Saving results...")
    analyzer.save_results()
    
    print("Generating summary report...")
    analyzer.generate_summary_report()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("Generated files:")
    print("- wheat_phenology_results.csv: Daily predictions with weather data")
    print("- wheat_phenology_analysis.png: Enhanced visualization")
    print()
    print("Key Features:")
    print("- Real weather data from Open-Meteo API")
    print("- Growing Degree Days (GDD) calculation")
    print("- Weather-informed growth stage estimation")
    print("- Ground cover percentage estimation")
    print("- Agricultural stress indices")
    print("- Location-specific analysis for Bavaria, Germany")

if __name__ == "__main__":
    main()

