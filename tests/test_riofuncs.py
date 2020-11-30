#!/usr/bin/env python3
"""
Unittests for the following functions in sungc/rasterio_funcs.py:
    * test 01 to 06 --> resample_bands()
    * test 07       --> check_image_singleval()
    * test 08       --> load_singleband()
    * test 09 to 11 --> load_bands()

Note:
    * resample_file_to_ds() is a wrapper for resample_band_to_ds()
      and won't be tested.

    * resample_band_to_ds() is a wrapper for rasterio.warp.reproject()
      and hence won't be tested

    * load_mask_from_shp() is indirectly tested in test_hedley_sung.py
      specify with,
      - test_fake_shp()
      - test_failure_point_shp()

"""
import pytest
import rasterio
import numpy as np

from pathlib import Path
from sungc import deglint
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling
import sungc.rasterio_funcs as rio_funcs
from . import check_resampled_array, check_resampled_metad

# specify the path to the odc_metadata.yaml of the test datasets
odc_meta_file = (
    Path(__file__).parent
    / "data/ga_ls8c_aard_3-2-0_091086_2014-11-06_final.odc-metadata.yaml"
)

# specify the product
product = "lmbadj"


def test_resample_1():
    """
    Ensures that rio_funcs.resample_bands() raises and Exception when:
        * resample_spatial_res <= 0
        * if save=True and odir=None
        * if save=True and odir does not exist.

    """
    g = deglint.GlintCorr(odc_meta_file, product)

    b3_file = g.band_list[g.band_ids.index("3")]
    with rasterio.open(b3_file, "r") as ds:
        spatial_res = ds.transform.a

    # ------------------------- #
    #   Raise Exception when    #
    # resample_spatial_res <= 0 #
    # ------------------------- #
    with pytest.raises(Exception) as excinfo:
        rio_funcs.resample_bands(
            bandlist=[b3_file],
            resample_spatial_res=0,
            resample_option=Resampling.nearest,
            load=True,
            save=False,
            odir=None,
        )
    assert "resample_spatial_res must be > 0" in str(excinfo)

    # ------------------------- #
    #   Raise Exception when    #
    #  save=True and odir=None  #
    # ------------------------- #
    with pytest.raises(Exception) as excinfo:
        rio_funcs.resample_bands(
            bandlist=[b3_file],
            resample_spatial_res=3 * spatial_res,
            resample_option=Resampling.nearest,
            load=False,
            save=True,
            odir=None,
        )
    assert "save requested for resampled geotiff, but odir not specified" in str(excinfo)

    # ------------------------------- #
    #       Raise Exception when      #
    #  save=True & not odir.exists()  #
    # ------------------------------- #
    non_path = Path("/freo/give/em/the/old/heave/ho")
    with pytest.raises(Exception) as excinfo:
        rio_funcs.resample_bands(
            bandlist=[b3_file],
            resample_spatial_res=3 * spatial_res,
            resample_option=Resampling.nearest,
            load=False,
            save=True,
            odir=non_path,
        )
    assert f"{non_path} does not exist" in str(excinfo)


def test_resample_2():
    """
    Ensures that the outputs from resample_bands() are as expected.
    In this test, load=True and save=False, meaning that the output
    values should be:
    resmpl_ofiles: NoneType (as save=False)
    spectral_cube: np.ndarray
        * ndim = 2 as ONE band is used.
        * same dtype as input band
        * dimensions 1/3 the size of input band
    metad: dict
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    downscale_factor = 3
    b3_file = g.band_list[g.band_ids.index("3")]
    with rasterio.open(b3_file, "r") as ds:
        ini_meta = ds.meta.copy()
        spatial_res = ini_meta["transform"].a

    resmpl_ofiles, resmpl_band, resmpl_metad = rio_funcs.resample_bands(
        bandlist=[b3_file],
        resample_spatial_res=downscale_factor * spatial_res,
        resample_option=Resampling.nearest,
        load=True,
        save=False,
        odir=None,
    )

    # check resmpl_ofiles
    assert resmpl_ofiles is None

    # check the resampled array
    check_resampled_array(resmpl_band, resmpl_metad, ini_meta, downscale_factor)

    # check the resampled metadata
    check_resampled_metad(resmpl_band, resmpl_metad, ini_meta)


def test_resample_3():
    """
    Ensures that the outputs from resample_bands() are as expected.
    In this test, load=True and save=False, meaning that the output
    values should be:
    resmpl_ofiles: NoneType (as save=False)
    spectral_cube: np.ndarray
        * ndim = 3 as a TWO bands is used.
        * same dtype as input band
        * dimensions 1/3 the size of input band
    metad: dict
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    downscale_factor = 3
    b2_file = g.band_list[g.band_ids.index("2")]
    b3_file = g.band_list[g.band_ids.index("3")]

    with rasterio.open(b3_file, "r") as ds:
        # in the test dataset, b2 and b3 have the same ds.meta
        ini_meta = ds.meta.copy()
        spatial_res = ini_meta["transform"].a

    resmpl_ofiles, resmpl_cube, resmpl_metad = rio_funcs.resample_bands(
        bandlist=[b2_file, b3_file],
        resample_spatial_res=downscale_factor * spatial_res,
        resample_option=Resampling.nearest,
        load=True,
        save=False,
        odir=None,
    )

    # check resmpl_ofiles
    assert resmpl_ofiles is None

    # check resampled output
    check_resampled_array(resmpl_cube, resmpl_metad, ini_meta, downscale_factor)

    # check the resampled metadata
    check_resampled_metad(resmpl_cube, resmpl_metad, ini_meta)


def test_resample_4(tmp_path):
    """
    Ensures that the outputs from resample_bands() are as expected.
    In this test, load=True and save=True, meaning that the output
    values should be:
    resmpl_ofiles: list, len=2
    spectral_cube: np.ndarray
        * ndim = 3 as a TWO bands is used.
        * same dtype as input band
        * dimensions 1/3 the size of input band
    metad: dict
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    downscale_factor = 3
    b2_file = g.band_list[g.band_ids.index("2")]
    b3_file = g.band_list[g.band_ids.index("3")]

    with rasterio.open(b3_file, "r") as ds:
        # in the test dataset, b2 and b3 have the same ds.meta
        ini_meta = ds.meta.copy()
        spatial_res = ini_meta["transform"].a

    resmpl_dir = tmp_path / "resample_test"
    resmpl_dir.mkdir()

    resmpl_ofiles, resmpl_cube, resmpl_metad = rio_funcs.resample_bands(
        bandlist=[b2_file, b3_file],
        resample_spatial_res=downscale_factor * spatial_res,
        resample_option=Resampling.nearest,
        load=True,
        save=True,
        odir=resmpl_dir,
    )

    # check resmpl_ofiles
    assert isinstance(resmpl_ofiles, list)
    assert len(resmpl_ofiles) == 2
    for resmpl_tiff in resmpl_ofiles:
        assert isinstance(resmpl_tiff, Path)
        assert resmpl_tiff.suffix == ".tif"
        assert resmpl_tiff.exists()

    check_resampled_array(resmpl_cube, resmpl_metad, ini_meta, downscale_factor)
    check_resampled_metad(resmpl_cube, resmpl_metad, ini_meta)


def test_resample_5(tmp_path):
    """
    Ensures that the outputs from resample_bands() are as expected.
    In this test, load=False and save=True, meaning that the output
    values should be:
    resmpl_ofiles: list, len=2
    spectral_cube: NoneType
    metad: NoneType
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    downscale_factor = 3
    b2_file = g.band_list[g.band_ids.index("2")]
    b3_file = g.band_list[g.band_ids.index("3")]

    with rasterio.open(b3_file, "r") as ds:
        # in the test dataset, b2 and b3 have the same ds.meta
        spatial_res = ds.meta["transform"].a

    resmpl_dir = tmp_path / "resample_test"
    resmpl_dir.mkdir()

    resmpl_ofiles, resmpl_cube, resmpl_metad = rio_funcs.resample_bands(
        bandlist=[b2_file, b3_file],
        resample_spatial_res=downscale_factor * spatial_res,
        resample_option=Resampling.nearest,
        load=False,
        save=True,
        odir=resmpl_dir,
    )

    # check resmpl_ofiles
    assert isinstance(resmpl_ofiles, list)
    assert len(resmpl_ofiles) == 2
    for resmpl_tiff in resmpl_ofiles:
        assert isinstance(resmpl_tiff, Path)
        assert resmpl_tiff.suffix == ".tif"
        assert resmpl_tiff.exists()

    assert isinstance(resmpl_cube, type(None))
    assert isinstance(resmpl_metad, type(None))


def test_resample_6(tmp_path):
    """
    Ensures that the outputs from resample_bands() are as expected.
    In this test, load=False and save=True, meaning that the output
    values should be:
    resmpl_ofiles: list, len=1
    spectral_cube: NoneType
    metad: NoneType
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    downscale_factor = 3
    b3_file = g.band_list[g.band_ids.index("3")]

    with rasterio.open(b3_file, "r") as ds:
        spatial_res = ds.meta["transform"].a

    resmpl_dir = tmp_path / "resample_test"
    resmpl_dir.mkdir()

    resmpl_ofiles, resmpl_cube, resmpl_metad = rio_funcs.resample_bands(
        bandlist=[b3_file],
        resample_spatial_res=downscale_factor * spatial_res,
        resample_option=Resampling.nearest,
        load=False,
        save=True,
        odir=resmpl_dir,
    )

    # check resmpl_ofiles
    assert isinstance(resmpl_ofiles, list)
    assert len(resmpl_ofiles) == 1
    assert isinstance(resmpl_ofiles[0], Path)
    assert resmpl_ofiles[0].suffix == ".tif"
    assert resmpl_ofiles[0].exists()

    assert isinstance(resmpl_cube, type(None))
    assert isinstance(resmpl_metad, type(None))


def test_singleval_7():
    """
    test the rio_funcs.check_image_singleval() function with 2D array
    """
    for nodata in [-999, np.nan]:
        test_arr = np.zeros([100, 100], order="C", dtype=np.float32)
        test_arr.fill(nodata)

        with pytest.raises(Exception) as excinfo:
            rio_funcs.check_image_singleval(test_arr, nodata, "test_arr")
        assert "only contains a single value" in str(excinfo)


def test_load_singleband_8():
    """
    test load_singleband() function
    """
    g = deglint.GlintCorr(odc_meta_file, product)
    b3_file = g.band_list[g.band_ids.index("3")]

    im, meta = rio_funcs.load_singleband(b3_file)

    # check for consistency between im and it's metadata dict
    assert isinstance(im, np.ndarray)
    assert isinstance(meta, dict)
    assert isinstance(meta["crs"], CRS)
    assert isinstance(meta["transform"], Affine)
    assert im.ndim == 2
    assert im.shape[0] == meta["height"]
    assert im.shape[1] == meta["width"]
    assert im.dtype == np.dtype(meta["dtype"])
    assert meta["count"] == 1
    assert meta["driver"] == "GTiff"
    assert meta["nodata"] == -999.0
    assert not np.all(im == meta["nodata"])


def test_load_bands_9():
    """
    test the scale_factor input of load_bands()
    """
    with pytest.raises(Exception) as excinfo:
        rio_funcs.load_bands([], -1, True)
    assert "load_bands: scale_factor <= 0" in str(excinfo)


def test_load_bands_10():
    """
    test the outputs of load_bands()
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    b2_file = g.band_list[g.band_ids.index("2")]
    b3_file = g.band_list[g.band_ids.index("3")]
    b4_file = g.band_list[g.band_ids.index("4")]

    spectral_cube, meta = rio_funcs.load_bands(
        [b2_file, b3_file, b4_file], g.scale_factor, False
    )

    assert isinstance(spectral_cube, np.ndarray)
    assert isinstance(meta, dict)
    assert isinstance(meta["crs"], CRS)
    assert isinstance(meta["transform"], Affine)
    assert spectral_cube.ndim == 3
    assert spectral_cube.shape[0] == meta["count"]
    assert spectral_cube.shape[1] == meta["height"]
    assert spectral_cube.shape[2] == meta["width"]
    assert spectral_cube.dtype == np.dtype(meta["dtype"])
    for z in range(spectral_cube.shape[0]):
        assert not np.all(spectral_cube[z, :, :] == meta["nodata"])


def test_load_bands_11():
    """
    test that scaling actually occurs in load_bands()
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    b2_file = g.band_list[g.band_ids.index("2")]

    scaled_im, meta = rio_funcs.load_bands([b2_file], g.scale_factor, True)
    unscaled_im, meta = rio_funcs.load_bands([b2_file], g.scale_factor, False)

    scaled_im = np.array(scaled_im[0, :, :], copy=True, order="K")
    unscaled_im = np.array(unscaled_im[0, :, :], copy=True, order="K", dtype=np.float32)

    ix = (unscaled_im > 0) & (unscaled_im <= g.scale_factor)
    ratio = unscaled_im[ix] / scaled_im[ix]  # e.g. 834/0.0834 = 10,000
    ratio_mean = ratio.mean()
    range_about_mean = 100.0 * (ratio.max() - ratio.min()) / ratio_mean

    # Due to rounding errors, associated with converting to np.float32
    # during the scaling, we can't expect the ratio of all pixels to
    # equal g.scale_factor. Hence test that the mean == g.scale_factor
    # and that percentage range about the mean <= 0.0001.
    # A value of 0.0001 for a scale_factor of 10,000 implies that:
    # ratio.min()  =  9999.99
    # ratio.mean() = 10000.00
    # ratio.max()  = 10000.01
    assert np.all(ratio_mean == g.scale_factor)
    assert range_about_mean <= 0.0001
