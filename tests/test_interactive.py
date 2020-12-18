#!/usr/bin/env python3
"""
Unittests for the interactive module that allows a user to select a polygon ROI
"""
import pytest
import numpy as np

from sungc.interactive import RoiSelector


def test_interactive_failure():
    with pytest.raises(ValueError) as excinfo:
        RoiSelector(rgb_im=None)
    assert "rgb_im is not an np.ndarray" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:
        RoiSelector(rgb_im=np.arange(3 * 4).reshape(3, 4))  # dims = [3,4]
    assert "rgb_im np.ndarray is not 3-dimensional" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:
        RoiSelector(rgb_im=np.arange(3 * 4 * 5).reshape(3, 4, 5))  # dims = [3,4,5]
    assert "last dimension of rgb_im != 3" in str(excinfo)

    rgb_im = np.arange(6 * 7 * 3).reshape(6, 7, 3)
    with pytest.raises(ValueError) as excinfo:
        RoiSelector(rgb_im=rgb_im, max_ds=0)
    assert "max_ds in RoiSelector must be > 0" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:
        RoiSelector(rgb_im=rgb_im, max_ds=None)
    assert "max_ds in RoiSelector must be int or float" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:
        RoiSelector(rgb_im=rgb_im, poly_xy=((0, 0), (1, 1)))
    assert "input ploy_xy in RoiSelector must have >= 3 vertices" in str(excinfo)
