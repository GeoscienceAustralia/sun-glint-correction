Python module for applying sun glint corrections to optical reflectance image data.
The following three commonly used sunglint correction algorithms are supported:
1. Cox and Munk (1954) statistical/geometry approach;
2. Hedley et al. (2005) correlation approach, and;
3. NIR/SWIR subtraction approach (e.g. Dierssen et al., 2015)

Please go through the usage examples (below) as they contain handy hints

Notes
- This module has been tested on image data from Landsat-5, -7 and -8,
and Sentinel-2A/B. Further testing is required on WV2 data.
- Cox and Munk approach requires an input wind speed (not yet automated)
- The Hedley approach requires a user-selected region of interest.
This module creates an interactive matplotlib figure that a user can
manipulate (add/delete vertices) to create the desiged polygon over
the region of interest.

Comments on deglinting S2A/B using Hedley et al. (2005) and NIR subtraction:
- This module can use band 8 (10 m) to deglint bands 1, 2, 3, 4, 5 and 6,
that have varying spatial resolutions. In this case band 8 will be downscaled
to 60 m (for band 1) or 20 m (for bands 5 and 6). Here, rasterio is used for
the resampling (bilinear interpolation). However, artefacts in the in the
deglinted bands 5, 6 have been found when band 8 (10 m) is used rather band 7
or 8A (20 m) as the NIR band. Further work will be required to investigate
the best practice of downscaling higher resolution bands.

References
- Cox, C., Munk, W. 1954. Statistics of the Sea Surface Derived
from Sun Glitter. J. Mar. Res., 13, 198-227.

- Cox, C., Munk, W. 1954. Measurement of the Roughness of the Sea
Surface from Photographs of the Suns Glitter. J. Opt. Soc. Am.,
44, 838-850.

- Dierssen, H.M., Chlus, A., Russell, B. 2015. Hyperspectral
discrimination of floating mats of seagrass wrack and the
macroalgae Sargassum in coastal waters of Greater Florida
Bay using airborne remote sensing. Remote Sens. Environ.,
167(15), 247-258, doi: https://doi.org/10.1016/j.rse.2015.01.027

- Hedley, J. D., Harborne, A. R., Mumby, P. J. (2005). Simple and
robust removal of sun glint for mapping shallow-water benthos.
Int. J. Remote Sens., 26(10), 2107-2112.

## Installation
### NCI
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
### Laptop/Desktop

```python setup.py install --prefix=<install-dir>```
Here, `--prefix` is optional.

## Operating System tested

Linux

## Supported Satellites and Sensors
* Sentinel-2A/B
* Landsat-5 TM
* Landsat-7 ETM
* Landsat-8 OLI
* Worldview-2

## Usage
### Deglint Sentinel-2B MSI data
Perform the various sunglint corrections on Sentinel-2B data.
This code works on S2A and S2B.
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
# until wagl-water-atcor products become available,
# test with nbart products.
product = "nbart"

dc_name = "s2b_ard_granule"

# Rottnest - Perth, WA
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
    # sub_folderName = SYYYY_MM_DD
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
    # deglint Bands 1, 2, 3, 4, 5 and 6 with Cox and Munk (1954).
    # you need to provide a wind-speed (m/s), here its been
    # set to 5 m/s
    cm_bandList = g.cox_munk(
        vis_band_ids=["1", "2", "3", "4", "5", "6"],
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

    # 3) deglint the 20- and 60- m bands using band 8A (20 m)
    deglint_BList_other = g.hedley_2005(
        vis_band_ids=["1", "5", "6"],
        nir_band_id="8A",
        odir=hedley_dir,
        roi_shpfile=shp_file,
        plot=True,
    )
    # In Hedley et al. (2019), band 9 was used to deglint band 1
    # However, band 9 isn't currently distruibuted because it's
    # affected by water vapour. As such band 8A (20 m), in this
    # example, is downscaled to 60 m and used to deglint Band 1.

    # ---------------------- #
    #     NIR SUBTRACTION    #
    # ---------------------- #
    # deglint Bands 2, 3, and 4 by subtracting Band 8
    mnir_bandList_10m = g.nir_subtraction(
        vis_band_ids=["2", "3", "4"], nir_band_id="8", odir=mnir_dir,
    )
    # the deglinted bands are saved as geotifs in mnir_dir,
    # see mnir_bandList_10m

    # deglint bands 1, 5 and 6 using band 8A
    mnir_bandList_other = g.nir_subtraction(
        vis_band_ids=["1", "5", "6"], nir_band_id="8A", odir=mnir_dir,
    )

if __name__ == "__main__":
    main()
```

### Deglint Landsat-8 OLI
Perform the various sunglint corrections on Landsat-8 OLI data.
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
# until wagl-water-atcor products become available,
# test with nbart products.
product = "nbart"

dc_name = "ga_ls8c_ard_3"

# Rottnest - Perth, WA
avail_datasets = dc.find_datasets(
    product=dc_name,
    lon=(115.40, 115.80),
    lat=(-31.60, -32.40),
    time=("2020-02-02", "2020-02-04")
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
    # deglint Bands 1, 2, 3 and 4 with Cox and Munk (1954).
    # you need to provide a wind-speed (m/s), here its been
    # set to 5 m/s
    cm_bandList = g.cox_munk(
        vis_band_ids=["1", "2", "3", "4"],
        odir=cox_munk_dir,
        wind_speed=5,
    )
    # the deglinted bands are saved as geotifs in cox_munk_dir
    # see cm_bandList

    # ---------------------- #
    #  Hedley et al. (2019)  #
    # ---------------------- #
    # deglint Bands 1, 2, 3 and 4 using Hedley et al. (2005)

    # 1) create shapefile with an interactive matplotlib figure
    shp_file = pjoin(hedley_dir, g.obase_shp)
    if not os.path.exists(shp_file):
        g.create_roi_shp(shp_file=shp_file, dwnscaling_factor=10)

    # 2) deglint the 30 m bands using band 6, and plot correlations
    deglint_BList = g.hedley_2005(
        vis_band_ids=["1", "2", "3", "4"],
        nir_band_id="6",
        odir=hedley_dir,
        roi_shpfile=shp_file,
        plot=True,
    )
    # the deglinted bands are saved as geotifs in hedley_dir,
    # see deglint_BList. The correlation plots are also
    # saved in hedley_dir

    # ---------------------- #
    #     NIR SUBTRACTION    #
    # ---------------------- #
    # deglint Bands 1, 2, 3, and 4 by subtracting Band 6
    mnir_bandList = g.nir_subtraction(
        vis_band_ids=["1", "2", "3", "4"], nir_band_id="6", odir=mnir_dir,
    )
    # the deglinted bands are saved as geotifs in mnir_dir,
    # see mnir_bandList

if __name__ == "__main__":
    main()
```

## Pre-commit setup


A [pre-commit](https://pre-commit.com/) config is provided to automatically format
and check your code changes. This allows you to immediately catch and fix
issues before you raise a failing pull request (which run the same checks under
Travis).

If you don't use Conda, install pre-commit from pip:

    pip install pre-commit

If you do use Conda, install from conda-forge (*required* because the pip
version uses virtualenvs which are incompatible with Conda's environments)

    conda install pre_commit

Now install the pre-commit hook to the current repository:

    pre-commit install

Your code will now be formatted and validated before each commit. You can also
invoke it manually by running `pre-commit run`
