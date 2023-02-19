# sptv_code
Code used to generate synthetic particle tracks using PIV velocity data.

# Usage
Used for processing analog sandbox tectonic models. Uses velocimetry data from the MATLAB-based PIV toolkit PIVLab.

# Methods
Uses a k-d tree based method for determining the displacement of synthetically-generated particles that flow with the measured velocity field. Additionally measures the deformation front of analog wedges for a front-based spatial reference system.
Also allows for the measurement of model subsurface temperatures and pressures using methods detailed in [Thouvenin (2022)](https://hammer.purdue.edu/articles/thesis/THE_IMPACT_OF_EROSION_ON_EXHUMATION_AND_STRUCTURAL_CONFIGURATION_IN_MOUNTAIN_BELTS_INSIGHTS_FROM_IMAGE_VELOCIMETRY_ANALYSIS_OF_COULOMB_WEDGE_MODELS/20371848).

https://user-images.githubusercontent.com/112208915/219909601-5bee318a-a077-4be1-a62c-b05de2c8da4e.mp4

# Setup and Run

 1. Download dataset from [here](link-to-data-on-dropbox-or-whatever)
 2. place the `pjt_highfric_15deg_1cmglassbeads_9ero` directory into the `data/` directory
 3. Make sure to install prerequisites:
    ```pip install numpy pandas matplotlib scipy pims```
 4. Run `python main.py`
