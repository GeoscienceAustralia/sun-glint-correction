# coding=utf-8
"""
Module
"""
import yaml
import rasterio
import numpy as np

from pathlib import Path
from rasterio import DatasetReader
from rasterio.warp import Resampling

import sungc.rasterio_funcs as rio_funcs


def load_yaml_file(p: Path):
    with p.open() as f:
        contents = yaml.load(f, Loader=yaml.FullLoader)
    return contents


def urd(arr1, arr2, nodata):
    """
    Compute the unbaised relative difference between two arrays
    urd = 100 % * abs(arr1 - arr2)/(0.5*(arr1 + arr2))

    Note: urd is a 1-D array
    """
    ave_arr = 0.5 * (arr1 + arr2)
    valid_ix = (arr1 != nodata) & (arr2 != nodata) & (ave_arr != 0.0)

    urd_arr = 100.0 * np.abs(arr1[valid_ix] - arr2[valid_ix]) / ave_arr[valid_ix]
    return urd_arr


def assert_with_expected(gen_impath: Path, exp_impath: Path, tol: float):
    with rasterio.open(gen_impath, "r") as gen_ds, rasterio.open(
        exp_impath, "r"
    ) as exp_ds:

        # load the generated dataset
        gen_ds: DatasetReader
        gen_arr = gen_ds.read(1)

        # load the expected datasets
        exp_arr = exp_ds.read(1)

        assert gen_ds.nodata == -999.0
        assert gen_ds.driver == "GTiff"
        assert gen_ds.dtypes == ("int16",)

        assert gen_ds.count == exp_ds.count
        assert gen_ds.height == exp_ds.height
        assert gen_ds.width == exp_ds.width

        # Verify the number of nodata pixels.
        gen_num_nodata = len(np.where(gen_arr == gen_ds.nodata)[0])
        exp_num_nodata = len(np.where(exp_arr == exp_ds.nodata)[0])
        assert gen_num_nodata == exp_num_nodata

        # ensure that all valid sungint corrected pixels match expected
        urd_sungc = urd(gen_arr, exp_arr, gen_ds.nodata)

        # ensure that the max urd is < 0.001%
        assert urd_sungc.max() < tol


def create_halved_band(input_bandfile: Path, out_path: Path):

    # reducing the spatial resolution by a factor of two
    with rasterio.open(input_bandfile, "r") as ds:
        spatial_res = 2 * float(ds.transform.a)

    resmpl_tifs, refl_im, rio_meta = rio_funcs.resample_bands(
        [input_bandfile],
        spatial_res,
        Resampling.nearest,
        load=False,
        save=True,
        odir=out_path,
    )

    return resmpl_tifs, rio_meta


def check_resampled_array(
    resmpl_array: np.ndarray,
    resmpl_metad: dict,
    ini_metad: dict,
    downscale_factor: int,
):
    exp_ndim = None
    if resmpl_metad["count"] == 1:
        exp_ndim = 2
    if resmpl_metad["count"] > 1:
        exp_ndim = 3

    assert isinstance(resmpl_array, np.ndarray)
    assert resmpl_array.ndim <= 3
    assert resmpl_array.ndim >= 2
    assert resmpl_array.ndim == exp_ndim
    assert resmpl_array.dtype == np.dtype(ini_metad["dtype"])
    assert resmpl_array.dtype == np.dtype(resmpl_metad["dtype"])
    if resmpl_array.ndim == 2:
        assert resmpl_array.shape[0] == ini_metad["height"] // downscale_factor
        assert resmpl_array.shape[1] == ini_metad["width"] // downscale_factor

    if resmpl_array.ndim == 3:
        assert resmpl_array.shape[0] == resmpl_metad["count"]
        assert resmpl_array.shape[1] == ini_metad["height"] // downscale_factor
        assert resmpl_array.shape[2] == ini_metad["width"] // downscale_factor


def check_resampled_metad(resmpl_array: np.ndarray, resmpl_metad: dict, ini_metad: dict):
    assert isinstance(resmpl_metad, dict)
    assert resmpl_array.dtype == np.dtype(resmpl_metad["dtype"])
    assert resmpl_metad["crs"] == ini_metad["crs"]
    assert resmpl_metad["driver"] == ini_metad["driver"]
    assert resmpl_metad["nodata"] == ini_metad["nodata"]
    assert resmpl_metad["transform"].b == ini_metad["transform"].b
    assert resmpl_metad["transform"].c == ini_metad["transform"].c
    assert resmpl_metad["transform"].d == ini_metad["transform"].d
    assert resmpl_metad["transform"].f == ini_metad["transform"].f

    if resmpl_array.ndim == 2:
        assert resmpl_array.shape[0] == resmpl_metad["height"]
        assert resmpl_array.shape[1] == resmpl_metad["width"]
        assert resmpl_metad["count"] == 1

    if resmpl_array.ndim == 3:
        assert resmpl_array.shape[1] == resmpl_metad["height"]
        assert resmpl_array.shape[2] == resmpl_metad["width"]
        assert resmpl_metad["count"] == resmpl_array.shape[0]
