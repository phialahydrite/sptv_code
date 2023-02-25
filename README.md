# sptv_code
Code used to generate synthetic particle tracks using PIV velocity data.

# Usage
Used for processing analog sandbox tectonic models. Uses velocimetry data from the MATLAB-based PIV toolkit PIVLab.

# Methods
Uses a k-d tree based method for determining the displacement of synthetically-generated particles that flow with the measured velocity field. Additionally measures the deformation front of analog wedges for a front-based spatial reference system.
Also allows for the measurement of model subsurface temperatures and pressures using methods detailed in [Thouvenin (2022)](https://hammer.purdue.edu/articles/thesis/THE_IMPACT_OF_EROSION_ON_EXHUMATION_AND_STRUCTURAL_CONFIGURATION_IN_MOUNTAIN_BELTS_INSIGHTS_FROM_IMAGE_VELOCIMETRY_ANALYSIS_OF_COULOMB_WEDGE_MODELS/20371848).

![index](https://user-images.githubusercontent.com/112208915/221377725-76bf12f8-8390-4f53-8b78-95069cf3d8fb.png)

# Setup and Run

 1. Collect model image and PIVLab-generated velocimetry data into 'data' directory
 2. Make sure to install prerequisites:
    ```pip install numpy pandas matplotlib scipy pims```
 3. Run `python main.py`
