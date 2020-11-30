#!/usr/bin/env python3
"""
Unittests for the interactive module that allows a user to select a polygon ROI
"""
import pytest
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.backends import backend_qt5agg

from sungc import deglint
from sungc.interactive import RoiSelector
from sungc.visualise import display_image

# specify the path to the odc_metadata.yaml of the test datasets
odc_meta_file = (
    Path(__file__).parent
    / "data/ga_ls8c_aard_3-2-0_091086_2014-11-06_final.odc-metadata.yaml"
)

# specify the product
product = "lmbadj"


def test_interactive_figure():
    """
    Test if an interative figure is generated when the ROI shapefile
    doesn't exist. This will follow the create_roi_shp() function
    in the GlintCorr class of sungc/deglint.py.

    Note that the unittest on g.quicklook_rgb() is performed in
    test_quicklook_rgb.py
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    # generate a quicklook
    rgb_im, rgb_meta = g.quicklook_rgb(dwnscale_factor=3)

    # let the user select a ROI from the quicklook RGB
    ax = display_image(rgb_im, None)
    assert isinstance(ax, matplotlib.axes.Axes)

    mc = RoiSelector(ax=ax)
    mc.interative()

    # Ensure that mc.canvas is the correct type.
    assert isinstance(mc.canvas, backend_qt5agg.FigureCanvasQTAgg)


def test_interactive_failure():
    with pytest.raises(Exception) as excinfo:
        RoiSelector(ax=None)
    assert "ax in RoiSelector must be matplotlib.axes.Axes" in str(excinfo)

    fig, axes = plt.subplots(nrows=2, ncols=1)
    # Here axes -> numpy.ndarray and will therefore raise an Exception
    with pytest.raises(Exception) as excinfo:
        RoiSelector(ax=axes)
    assert "ax in RoiSelector must be matplotlib.axes.Axes" in str(excinfo)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    with pytest.raises(Exception) as excinfo:
        RoiSelector(ax=axes, max_ds=0)
    assert "max_ds in RoiSelector must be > 0" in str(excinfo)

    with pytest.raises(Exception) as excinfo:
        RoiSelector(ax=axes, max_ds=None)
    assert "max_ds in RoiSelector must be int or float" in str(excinfo)

    with pytest.raises(Exception) as excinfo:
        RoiSelector(ax=axes, poly_xy=((0, 0), (1, 1)))
    assert "input ploy_xy in RoiSelector must have >= 3 vertices" in str(excinfo)
