# Spatio-temporal & Epidemiological Analysis Module

This module provides tools for spatial and epidemiological analysis of antibiotic resistance data.

## Overview

The `spatiotemporal` module consists of four main components:

1. **SpatialAnalyzer** - Spatial clustering and hotspot detection
2. **TemporalAnalyzer** - Time series analysis and trend detection  
3. **SourceAttributor** - Source attribution and comparison
4. **SpatioTemporalVisualizer** - Visualization and dashboard generation

## Installation

Required packages:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

## Usage

### Complete Analysis

Run the full Phase 5 analysis:

```bash
python phase5_spatiotemporal.py
```

This will perform:
- Spatial clustering and hotspot identification
- Temporal trend analysis (if date/time data available)
- Source attribution and comparison
- Comprehensive visualizations

### Individual Components

#### 1. Spatial Analysis

```python
from spatiotemporal import SpatialAnalyzer

# Initialize
analyzer = SpatialAnalyzer(min_samples=5, eps_km=50)

# Prepare data
df_spatial = analyzer.prepare_data(df)

# Geographic clustering
clusters = analyzer.cluster_by_geography(df_spatial)

# Identify hotspots (75th percentile threshold)
hotspots = analyzer.identify_hotspots(df_spatial, threshold_percentile=75)

# Compute spatial statistics
stats = analyzer.compute_spatial_statistics(df_spatial, by_species=True, by_source=True)

# Save results
analyzer.save_results(output_prefix='spatial_analysis')
```

**Methods:**
- `cluster_by_geography()` - Hierarchical clustering by administrative region, site, location
- `identify_hotspots()` - Detect high-resistance locations above threshold percentile
- `compute_spatial_statistics()` - Calculate resistance prevalence by region/species/source
- `generate_hotspot_report()` - Generate comprehensive hotspot analysis

**Outputs:**
- `spatial_analysis_hotspots.json` - Hotspot locations per antibiotic
- `spatial_analysis_clusters.json` - Geographic cluster profiles
- `spatial_analysis_stats.json` - Spatial statistics

#### 2. Temporal Analysis

```python
from spatiotemporal import TemporalAnalyzer

# Initialize (30-day window, 1.5x threshold for alerts)
analyzer = TemporalAnalyzer(window_size=30, alert_threshold=1.5)

# Prepare temporal data
df_temporal = analyzer.prepare_temporal_data(df, date_column='collection_date')

if df_temporal is not None:
    # Rolling prevalence
    time_series = analyzer.compute_rolling_prevalence(
        df_temporal, 
        date_column='collection_date',
        by_species=True
    )
    
    # Detect change points
    change_points = analyzer.detect_change_points(min_change=10)
    
    # Analyze trends
    trends = analyzer.analyze_trends()
    
    # Generate alerts
    alerts = analyzer.generate_alerts()
    
    # Save results
    analyzer.save_results(output_prefix='temporal_analysis')
```

**Methods:**
- `compute_rolling_prevalence()` - Calculate rolling resistance rates
- `detect_change_points()` - Identify significant shifts in resistance
- `analyze_trends()` - Linear trend analysis per antibiotic
- `generate_alerts()` - Flag significant increases for intervention

**Outputs:**
- `temporal_analysis_report.json` - Trends, change points, alerts
- `temporal_analysis_timeseries.csv` - Rolling prevalence data

**Note:** Temporal analysis requires a date/time column in the data. The framework is ready for use when temporal data becomes available.

#### 3. Source Attribution

```python
from spatiotemporal import SourceAttributor

# Initialize
attributor = SourceAttributor()

# Prepare data
df_source = attributor.prepare_data(df)

# Analyze by source
source_profiles = attributor.analyze_by_source(df_source)

# Statistical comparison (Chi-square test)
comparisons = attributor.compare_sources(df_source, test='chi2')

# Identify reservoirs (75th percentile)
reservoirs = attributor.identify_reservoirs(df_source, threshold_percentile=75)

# Save results
attributor.save_results(output_prefix='source_attribution')
```

**Methods:**
- `analyze_by_source()` - Calculate resistance profiles per sample source
- `compare_sources()` - Statistical comparison between sources (chi-square or Fisher's exact)
- `identify_reservoirs()` - Detect potential environmental reservoirs
- `generate_attribution_report()` - Comprehensive source analysis

**Outputs:**
- `source_attribution_report.json` - Source profiles, comparisons, reservoirs

#### 4. Visualization

```python
from spatiotemporal import SpatioTemporalVisualizer

# Initialize
visualizer = SpatioTemporalVisualizer(output_dir='visualizations')

# Regional heatmap
visualizer.plot_regional_heatmap(df, antibiotics=None)

# Source comparison
if source_profiles:
    visualizer.plot_source_comparison(source_profiles)

# Temporal trends
if time_series is not None:
    visualizer.plot_temporal_trends(time_series)

# Hotspot maps
if hotspots:
    visualizer.plot_hotspot_map(hotspots, level='regional')
    visualizer.plot_hotspot_map(hotspots, level='site')

# Comprehensive dashboard
visualizer.create_dashboard(
    df,
    spatial_results=clusters,
    temporal_results=time_series,
    source_results=source_profiles
)

# Or generate all plots at once
visualizer.save_all_plots(df, spatial_analyzer, temporal_analyzer, source_attributor)
```

**Visualization Types:**
- Regional heatmap - Resistance by region and antibiotic
- Source comparison - Bar plots comparing sample sources
- Temporal trends - Time series for resistance over time
- Hotspot maps - Geographic hotspot visualization
- Dashboard - Comprehensive multi-panel overview

**Outputs:** All visualizations saved as PNG files (300 DPI)

## Data Requirements

### Required Columns

**Spatial Analysis:**
- `administrative_region` - Administrative region identifier
- `national_site` - Site identifier (optional)
- `local_site` - Fine-grained location (optional)
- `{antibiotic}_int` - Antibiotic interpretation (s/i/r)

**Temporal Analysis:**
- Date/time column (e.g., `collection_date`, `date`)
- `{antibiotic}_int` - Antibiotic interpretation
- `bacterial_species` - Species name (optional, for stratified analysis)

**Source Attribution:**
- `sample_source` - Sample source identifier (required)
- `{antibiotic}_int` - Antibiotic interpretation
- `bacterial_species` - Species name (optional)
- `esbl` - ESBL status (optional)

## Output Structure

```
results/phase5/
├── spatial_analysis_hotspots.json
├── spatial_analysis_clusters.json
├── spatial_analysis_stats.json
├── temporal_analysis_report.json
├── temporal_analysis_timeseries.csv
├── source_attribution_report.json
├── phase5_summary.json
└── visualizations/
    ├── regional_heatmap.png
    ├── source_comparison.png
    ├── temporal_trends.png
    ├── hotspot_map_regional.png
    ├── hotspot_map_site.png
    └── dashboard.png
```

## Interpretation Guide

### Hotspot Detection

Hotspots are locations with resistance prevalence above the threshold percentile (default: 75th).

- **High-priority hotspots** - Locations with resistance ≥ 90%
- **Medium-priority hotspots** - Locations with resistance 75-90%
- **Monitoring zones** - Locations approaching threshold

**Actions:**
- Increase surveillance in hotspot areas
- Investigate local prescribing patterns
- Implement targeted interventions

### Temporal Alerts

Alerts are generated when resistance increases by ≥ 1.5x baseline.

- **High severity** - ≥2x increase
- **Medium severity** - 1.5-2x increase

**Actions:**
- Investigate cause of increase
- Review recent prescribing changes
- Implement antimicrobial stewardship

### Source Attribution

Statistical comparisons identify sources with significantly different resistance profiles.

- **p < 0.05** - Significant difference
- **p < 0.01** - Highly significant

**Reservoirs** are sources with consistently high resistance (≥75th percentile).

**Actions:**
- Enhanced hygiene in reservoir sources
- Source-specific interventions
- Environmental sampling programs

## Advanced Usage

### Custom Thresholds

```python
# Lower hotspot threshold (50th percentile)
hotspots = analyzer.identify_hotspots(df, threshold_percentile=50)

# More sensitive alerts (1.2x increase)
analyzer = TemporalAnalyzer(alert_threshold=1.2)

# Different reservoir threshold (90th percentile)
reservoirs = attributor.identify_reservoirs(df, threshold_percentile=90)
```

### Subset Analysis

```python
# Analyze specific species
df_ecoli = df[df['bacterial_species'] == 'e_coli']
hotspots_ecoli = analyzer.identify_hotspots(df_ecoli)

# Analyze specific region
df_region = df[df['administrative_region'] == 'region_i']
source_profiles_region = attributor.analyze_by_source(df_region)
```

### Export for GIS

```python
# Export hotspot coordinates for mapping software
import json

hotspot_data = []
for antibiotic, levels in hotspots.items():
    for level, data in levels.items():
        for location, rate in data['hotspots'].items():
            hotspot_data.append({
                'antibiotic': antibiotic,
                'location': location,
                'level': level,
                'resistance_%': rate
            })

with open('hotspots_for_gis.json', 'w') as f:
    json.dump(hotspot_data, f, indent=2)
```

## Troubleshooting

### "No temporal data available"

The temporal analysis requires a date/time column. If your data doesn't have temporal information:
- The framework is still installed and ready for future use
- Spatial and source analyses will work normally
- Add date collection when possible for trend analysis

### "No spatial columns found"

Ensure your data has at least one of:
- `administrative_region`
- `national_site`
- `local_site`

### "Insufficient data for statistical test"

Some source comparisons require minimum sample sizes (n≥5 per group). This is normal for rare sources.

### Memory Issues

For large datasets:
```python
# Limit number of antibiotics
analyzer.identify_hotspots(df, antibiotics=['ampicillin', 'ciprofloxacin'])

# Reduce visualization complexity
visualizer.plot_regional_heatmap(df, antibiotics=top_10_antibiotics)
```

## References

### Spatial Methods
- Kulldorff M. (1997). A spatial scan statistic. Communications in Statistics - Theory and Methods
- Anselin L. (1995). Local Indicators of Spatial Association - LISA

### Temporal Methods
- Taylor WA. (2000). Change-Point Analysis: A Powerful Tool For Detecting Changes
- Cleveland RB, et al. (1990). STL: A Seasonal-Trend Decomposition Procedure

### Statistical Tests
- Pearson K. (1900). On the criterion that a given system of deviations
- Fisher RA. (1922). On the interpretation of χ² from contingency tables

## Contact & Support

For issues or questions about this module, please refer to the main project documentation or create an issue in the repository.
