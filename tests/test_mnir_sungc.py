#!/usr/bin/env python3

"""
Unittests for the NIR subtraction algorithm
"""

import pytest
import rasterio
from pathlib import Path

from sungc import deglint
from . import urd

# specify the path to the odc_metadata.yaml of the test datasets
data_path = Path(__file__).parent / "data"
odc_meta_file = data_path / "ga_ls8c_aard_3-2-0_091086_2014-11-06_final.odc-metadata.yaml"

# specify the sub_product
sub_product = "lmbadj"


def test_mnir_image():
    """
    Check that the generated deglinted band is nearly identical
    to the expected deglinted band
    """
    # Initiate the sunglint correction class
    g = deglint.GlintCorr(odc_meta_file, sub_product)

    # ---------------------- #
    #     NIR subtraction    #
    # ---------------------- #
    mnir_xarrlist = g.glint_subtraction(
        vis_bands=["3"],
        corr_band="6",
        water_val=5,
    )

    sungc_band = mnir_xarrlist[0].lmbadj_green.values  # 3D array

    # path to expected sunglint corrected output from NIR subtraction
    exp_sungc_band = (
        data_path
        / "MINUS_NIR"
        / "ga_ls8c_lmbadj_3-2-0_091086_2014-11-06_final_band03-deglint-600m.tif"
    )

    # ensure that all valid sungint corrected pixels match expected
    with rasterio.open(exp_sungc_band, "r") as exp_sungc_ds:
        urd_band = urd(sungc_band[0, :, :], exp_sungc_ds.read(1), exp_sungc_ds.nodata)
        assert urd_band.max() < 0.001


def test_mnir_bands():
    """
    Ensure that glint_subtraction() raises and Exception if
    the specified vis_band_id/corr_band do not exist
    """
    g = deglint.GlintCorr(odc_meta_file, sub_product)

    with pytest.raises(Exception) as excinfo:
        g.glint_subtraction(
            vis_bands=["20"],  # this band id doesn't exist
            corr_band="6",
            water_val=5,
        )
    assert "is missing from bands" in str(excinfo)

    with pytest.raises(Exception) as excinfo:
        g.glint_subtraction(
            vis_bands=["3"],
            corr_band="20",  # this band id doesn't exist
            water_val=5,
        )
    assert "is missing from bands" in str(excinfo)


def test_empty_inputs():
    """
    Ensure that glint_subtraction() raises and Exception if
    the VIS and NIR band only contain nodata pixels
    """
    g = deglint.GlintCorr(odc_meta_file, sub_product)

    with pytest.raises(Exception) as excinfo:
        g.glint_subtraction(
            vis_bands=["3"],
            corr_band="7",  # this dummy band only contains nodata
            water_val=5,
        )
    assert "only contains a single value" in str(excinfo)

    with pytest.raises(Exception) as excinfo:
        g.glint_subtraction(
            vis_bands=["7"],  # this dummy band only contains nodata
            corr_band="6",
            water_val=5,
        )
    assert "only contains a single value" in str(excinfo)
