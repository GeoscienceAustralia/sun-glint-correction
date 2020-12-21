#!/usr/bin/env python3
"""
Unittests for the creation of quicklook RGB.
These functions are used in the Hedley et al. (2005)
implementation when an interative matplotlib figure
is generated.

Unittest include:
    * ensuring quicklook rgb has the correct dims, dtype etc.
"""
import pytest
import rasterio
import numpy as np
from pathlib import Path

from sungc import deglint
from sungc.rasterio_funcs import quicklook_rgb

# specify the path to the odc_metadata.yaml of the test datasets
odc_meta_file = (
    Path(__file__).parent
    / "data/ga_ls8c_aard_3-2-0_091086_2014-11-06_final.odc-metadata.yaml"
)

# specify the sub_product
sub_product = "lmbadj"


def test_quicklook_gen(tmp_path):
    """
    Test that the quicklook rgb is as expected
    """
    dwnscale_factor = 3
    g = deglint.GlintCorr(odc_meta_file, sub_product)

    # generate a quicklook
    rgb_im, rgb_meta = quicklook_rgb(
        rgb_bandlist=g.rgb_bandlist,
        scale_factor=g.scale_factor,
        dwnscale_factor=dwnscale_factor,
    )

    rgb_shape = rgb_im.shape

    assert isinstance(rgb_im, np.ndarray) is True
    assert rgb_im.ndim == 3
    assert rgb_shape[2] == 3
    assert rgb_im.dtype == np.uint8
    assert rgb_meta["band_1"] == "ga_ls8c_lmbadj_3-2-0_091086_2014-11-06_final_band04"
    assert rgb_meta["band_2"] == "ga_ls8c_lmbadj_3-2-0_091086_2014-11-06_final_band03"
    assert rgb_meta["band_3"] == "ga_ls8c_lmbadj_3-2-0_091086_2014-11-06_final_band02"

    ix_b3 = g.band_ids.index("3")
    # check that the downsampling procedure worked!
    with rasterio.open(g.band_list[ix_b3], "r") as b3_ds:
        b3_meta = b3_ds.meta.copy()
        assert rgb_meta["height"] == b3_meta["height"] // dwnscale_factor
        assert rgb_meta["width"] == b3_meta["width"] // dwnscale_factor
        assert rgb_meta["transform"].b == b3_meta["transform"].b
        assert rgb_meta["transform"].c == b3_meta["transform"].c
        assert rgb_meta["transform"].d == b3_meta["transform"].d
        assert rgb_meta["transform"].f == b3_meta["transform"].f
        assert rgb_meta["transform"].a == b3_meta["transform"].a * dwnscale_factor
        assert rgb_meta["transform"].e == b3_meta["transform"].e * dwnscale_factor

    # check Exception is raised if dwnscale_factor < 1
    with pytest.raises(ValueError) as excinfo:
        quicklook_rgb(
            rgb_bandlist=g.rgb_bandlist,
            scale_factor=g.scale_factor,
            dwnscale_factor=0.999,
        )
    assert "dwnscale_factor must be a float >= 1" in str(excinfo)
