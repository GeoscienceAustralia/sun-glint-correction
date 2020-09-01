Python module for applying sun glint corrections to optical reflectance image data.
The following three commonly used sunglint correction algorithms are supported:
1) Cox and Munk (1954) statistical/geometry approach;
2) Hedley et al. (2005) correlation approach, and;
3) NIR subtraction approach (e.g. Dierssen et al., 2015)

References:
Cox, C., Munk, W. 1954. Statistics of the Sea Surface Derived
    from Sun Glitter. J. Mar. Res., 13, 198-227.

Cox, C., Munk, W. 1954. Measurement of the Roughness of the Sea
    Surface from Photographs of the Suns Glitter. J. Opt. Soc. Am.,
    44, 838-850.

Dierssen, H.M., Chlus, A., Russell, B. 2015. Hyperspectral
    discrimination of floating mats of seagrass wrack and the
    macroalgae Sargassum in coastal waters of Greater Florida
    Bay using airborne remote sensing. Remote Sens. Environ.,
    167(15), 247-258, doi: https://doi.org/10.1016/j.rse.2015.01.027

Hedley, J. D., Harborne, A. R., Mumby, P. J. (2005). Simple and
    robust removal of sun glint for mapping shallow-water benthos.
    Int. J. Remote Sens., 26(10), 2107-2112.

## Installation

Log into NCI and clone this repository into a dir/workspace in you NCI home dir:

```BASH
cd ~/<your project dir>
git clone git@github.com:GeoscienceAustralia/sun-glint-correction.git
cd sun-glint-correction
git checkout -b develop

# set up a local Python 3.6 runtime environment
source configs/sunglint.env  # should be error/warning free
export CUSTOM_PY_INSTALL=~/.digitalearthau/dea-env/20200713/local
mkdir -p $CUSTOM_PY_INSTALL/lib/python3.6/site-packages/
python setup.py install --prefix=$CUSTOM_PY_INSTALL
```

## Operating System tested

Linux

## Supported Satellites and Sensors
* Sentinel-2A/B
* Landsat-5 TM
* Landsat-7 ETM
* Landsat-8 OLI
* Worldview-2

## Usage
### Deglint Sentinel-2B data
Perform the various sunglint corrections on Sentinel-2B data.
First source configs/sunglint.env

```python
#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from os.path import join as pjoin

import datacube
from sungc import deglint

dc = datacube.Datacube()
product = "nbart"  # test until water-atcor products become available

dc_name = "s2b_ard_granule"

avail_datasets = dc.find_datasets(
    product=dc_name,
    lon=(115.40, 115.80),
    lat=(-31.60, -32.40),
    time=("2019-01-31", "2019-02-02")
)

working_dir = "/path/to/somewhere/"

def main():

    # for this test select the first dataset
    ds = avail_datasets[0]

    g = deglint.GlintCorr(ds, product)

    # Create output directories
    # sub_folderName = LYYYY_MM_DD
    overpass_dt = g.overpass_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
    sub_folderName = (
        g.sensor[0].upper()
        + "_".join(overpass_dt.split()[0].split("-"))
    )

    mnir_dir = pjoin(working_dir, "MINUS_NIR", dc_name, sub_folderName)
    hedley_dir = pjoin(working_dir, "HEDLEY", dc_name, sub_folderName)
    cox_munk_dir = pjoin(working_dir, "COX_MUNK", dc_name, sub_folderName)

    # make output directories
    os.makedirs(mnir_dir, exist_ok=True)
    os.makedirs(hedley_dir, exist_ok=True)
    os.makedirs(cox_munk_dir, exist_ok=True)

    # ---------------------- #
    #  Cox and Munk (1954)   #
    # ---------------------- #
    # deglint Bands 2, 3 and 4 with Cox and Munk (1954).
    # you need to provide a wind-speed (m/s)
    cm_bandList = g.cox_munk(
        vis_band_ids=["2", "3", "4"],
        odir=cox_munk_dir,
        wind_speed=5,
    )
    # the deglinted bands are saved as geotifs in cox_munk_dir
    # see cm_bandList

    # ---------------------- #
    #  Hedley et al. (2019)  #
    # ---------------------- #
    # deglint Bands 2, 3 and 4 using Hedley et al. (2005)

    # 1) create shapefile with an interactive matplotlib figure
    shp_file = pjoin(hedley_dir, g.obase_shp)
    if not os.path.exists(shp_file):
        g.create_roi_shp(shp_file=shp_file, dwnscaling_factor=10)

    # 2) deglint the 10 m bands using band 8, and plot correlations
    deglint_BList_10m = g.hedley_2005(
        vis_band_ids=["2", "3", "4"],
        nir_band_id="8",
        odir=hedley_dir,
        roi_shpfile=shp_file,
        plot=True,
    )
    # the deglinted bands are saved as geotifs in hedley_dir,
    # see deglint_BList_10m. The correlation plots are also 
    # saved in hedley_dir

    # ---------------------- #
    #     NIR SUBTRACTION    #
    # ---------------------- #
    # deglint Bands 2, 3, and 4 by subtracting Band 8
    mnir_bandList = g.nir_subtraction(
        vis_band_ids=["2", "3", "4"], nir_band_id="8", odir=mnir_dir,
    )
    # the deglinted bands are saved as geotifs in mnir_dir,
    # see mnir_bandList
```
