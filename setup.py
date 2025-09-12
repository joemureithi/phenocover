#!/usr/bin/env python3
"""
Setup script for Weather-Enhanced Wheat Phenology Analysis Tool
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        "wheat_phenology_analyzer.py",
        "demo.py",
        "requirements.txt",
        "field_location.geojson",
        "NDVI_ Treatment Parcel - 0-data-2025-07-04 15_11_14.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("All required files present!")
        return True

def main():
    """Main setup function"""
    print("="*60)
    print("WEATHER-ENHANCED WHEAT PHENOLOGY ANALYSIS SETUP")
    print("="*60)
    
    # Check files
    if not check_files():
        print("Please ensure all required files are present.")
        return False
    
    # Install requirements
    if not install_requirements():
        print("Please install requirements manually: pip install -r requirements.txt")
        return False
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("You can now run the analysis:")
    print("  python demo.py")
    print()
    print("Or use the analyzer directly:")
    print("  from wheat_phenology_analyzer import WheatPhenologyAnalyzer")
    print()
    print("Features available:")
    print("  - Real weather data from Open-Meteo API")
    print("  - Weather-informed growth stage estimation")
    print("  - Ground cover percentage calculation")
    print("  - Agricultural stress indices")
    print("  - Enhanced visualizations")
    
    return True

if __name__ == "__main__":
    main()

