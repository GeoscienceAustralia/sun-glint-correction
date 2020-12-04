#!/usr/bin/env python3

"""
Unittests for the NIR subtraction algorithm
"""

import pytest
from pathlib import Path

from sungc import deglint
from . import assert_with_expected

# specify the path to the odc_metadata.yaml of the test datasets
data_path = Path(__file__).parent / "data"
odc_meta_file = data_path / "ga_ls8c_aard_3-2-0_091086_2014-11-06_final.odc-metadata.yaml"

# specify the product
product = "lmbadj"


def test_mnir_image(tmp_path):
    """
    Check that the generated deglinted band is nearly identical
    to the expected deglinted band
    """
    # Initiate the sunglint correction class
    g = deglint.GlintCorr(odc_meta_file, product)

    # ---------------------- #
    #     NIR subtraction    #
    # ---------------------- #
    mnir_dir = tmp_path / "MINUS_NIR"
    mnir_dir.mkdir()

    mnir_bandlist = g.nir_subtraction(
        vis_band_ids=["3"],
        nir_band_id="6",
        odir=mnir_dir,
        water_val=5,
    )

    # path to expected sunglint corrected output from NIR subtraction
    exp_sungc_band = (
        data_path
        / "MINUS_NIR"
        / "ga_ls8c_lmbadj_3-2-0_091086_2014-11-06_final_band03-deglint-600m.tif"
    )

    # assert that output in mnir_bandlist matches expected_sungc_band3
    assert_with_expected(Path(mnir_bandlist[0]), exp_sungc_band, 0.001)


def test_mnir_bands(tmp_path):
    """
    Ensure that nir_subtraction() raises and Exception if
    the specified vis_band_id/nir_band_id do not exist
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    mnir_dir = tmp_path / "MINUS_NIR"
    mnir_dir.mkdir()

    with pytest.raises(Exception) as excinfo:
        g.nir_subtraction(
            vis_band_ids=["20"],  # this band id doesn't exist
            nir_band_id="6",
            odir=mnir_dir,
            water_val=5,
        )
    assert "is missing from bands" in str(excinfo)

    with pytest.raises(Exception) as excinfo:
        g.nir_subtraction(
            vis_band_ids=["3"],
            nir_band_id="20",  # this band id doesn't exist
            odir=mnir_dir,
            water_val=5,
        )
    assert "is missing from bands" in str(excinfo)


def test_empty_inputs(tmp_path):
    """
    Ensure that nir_subtraction() raises and Exception if
    the VIS and NIR band only contain nodata pixels
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    mnir_dir = tmp_path / "MINUS_NIR"
    mnir_dir.mkdir()

    with pytest.raises(Exception) as excinfo:
        g.nir_subtraction(
            vis_band_ids=["3"],
            nir_band_id="7",  # this dummy band only contains nodata
            odir=mnir_dir,
            water_val=5,
        )
    assert "only contains a single value" in str(excinfo)

    with pytest.raises(Exception) as excinfo:
        g.nir_subtraction(
            vis_band_ids=["7"],  # this dummy band only contains nodata
            nir_band_id="6",
            odir=mnir_dir,
            water_val=5,
        )
    assert "only contains a single value" in str(excinfo)
