#!/usr/bin/env python3

import yaml
import pytest
from pathlib import Path

from . import load_yaml_file
from sungc import deglint

# specify the path to the odc_metadata.yaml of the test datasets
data_path = Path(__file__).parent / "data"
odc_meta_file = data_path / "ga_ls8c_aard_3-2-0_091086_2014-11-06_final.odc-metadata.yaml"

# specify the product
product = "lmbadj"


def test_glintcorr_1(tmp_path):
    """
    Ensure that GlintCorr will raise an Exception if
    ".odc-metadata.yaml" not in yaml basename
    """
    random_yaml = tmp_path / "random.yaml"
    with pytest.raises(Exception) as excinfo:
        deglint.GlintCorr(random_yaml, product)
    assert "is not a .odc-metadata.yaml" in str(excinfo)


def test_glintcorr_2(tmp_path):
    """
    Ensure that GlintCorr will raise an Exception if
    it couldn't find the "measurement" key in
    .odc-metadata.yaml. The dictionary contained in
    this key has the required band names.

    This will test GlintCorr.get_meas_dict()
    """
    # load the odc_meta_file
    meta_dict = load_yaml_file(odc_meta_file)

    # change "measurement" key
    meta_dict["image_data"] = meta_dict.pop("measurements")

    # save changes
    mod_meta_file = tmp_path / "ga_ls8c_aard_3-2-0_091086_MODIFIED.odc-metadata.yaml"
    with open(mod_meta_file, "w") as ofid:
        yaml.dump(meta_dict, ofid)

    with pytest.raises(Exception) as excinfo:
        deglint.GlintCorr(mod_meta_file, product)
    assert "neither image nor measurements keys were found in" in str(excinfo)


def test_glintcorr_3(tmp_path):
    """
    Ensure that GlintCorr will raise an Exception if
    it can't find the sensor information from the
    .odc-metadata.yaml.

    This will test GlintCorr.get_sensor()
    """
    # load the odc_meta_file
    meta_dict = load_yaml_file(odc_meta_file)

    # change "sensor" key
    meta_dict["properties"]["sensor_info"] = meta_dict["properties"].pop("eo:platform")

    # save changes

    mod_meta_file = tmp_path / "ga_ls8c_aard_3-2-0_091086_MODIFIED.odc-metadata.yaml"
    with open(mod_meta_file, "w") as ofid:
        yaml.dump(meta_dict, ofid)

    with pytest.raises(Exception) as excinfo:
        deglint.GlintCorr(mod_meta_file, product)
    assert "Unable to extract sensor name" in str(excinfo)


def test_glintcorr_4(tmp_path):
    """
    Ensure that GlintCorr will raise an Exception if
    the sensor in the .odc-metadata.yaml isn't supported.

    This will test GlintCorr.check_sensor()
    """
    # load the odc_meta_file
    meta_dict = load_yaml_file(odc_meta_file)

    # change sensor to something random
    meta_dict["properties"]["eo:platform"] = "DESIS-999"

    # save changes
    mod_meta_file = tmp_path / "ga_ls8c_aard_3-2-0_091086_MODIFIED.odc-metadata.yaml"
    with open(mod_meta_file, "w") as ofid:
        yaml.dump(meta_dict, ofid)

    with pytest.raises(Exception) as excinfo:
        deglint.GlintCorr(mod_meta_file, product)
    assert "Supported sensors are" in str(excinfo)


def test_glintcorr_5(tmp_path):
    """
    Ensure that GlintCorr will raise an Exception if
    GlintCorr is unable to extract the overpass
    datetime information from .odc-metadata.yaml

    This will test GlintCorr.get_overpass_datetime()
    """
    # load the odc_meta_file
    meta_dict = load_yaml_file(odc_meta_file)

    # change "datetime" key
    meta_dict["properties"]["overpass_time"] = meta_dict["properties"].pop("datetime")

    # save changes
    mod_meta_file = tmp_path / "ga_ls8c_aard_3-2-0_091086_MODIFIED.odc-metadata.yaml"
    with open(mod_meta_file, "w") as ofid:
        yaml.dump(meta_dict, ofid)

    with pytest.raises(Exception) as excinfo:
        deglint.GlintCorr(mod_meta_file, product)
    assert "Unable to extract overpass datetime" in str(excinfo)


def test_glintcorr_6():
    """
    Ensure that GlintCorr will raise an Exception if
    geotiffs associated with products that were not
    packaged/indexed are requested. Here, lmbadj
    products were packaged, however, lmbskyg bands
    are requested.

    This will test GlintCorr.get_band_list()
    """
    with pytest.raises(Exception) as excinfo:
        deglint.GlintCorr(odc_meta_file, "lmbskyg")
    assert "Could not find any geotifs in" in str(excinfo)


def test_glintcorr_7(tmp_path):
    """
    Ensure that GlintCorr will raise an Exception if
    the bands specified in .odc-metadata.yaml do not
    exists.

    This will test GlintCorr.get_band_list()
    """
    # load the odc_meta_file
    meta_dict = load_yaml_file(odc_meta_file)

    # save to tmp_path where the bands won't exist
    mod_meta_file = tmp_path / "ga_ls8c_aard_3-2-0_091086_MODIFIED.odc-metadata.yaml"
    with open(mod_meta_file, "w") as ofid:
        yaml.dump(meta_dict, ofid)

    band2 = tmp_path / meta_dict["measurements"]["lmbadj_blue"]["path"]
    with pytest.raises(Exception) as excinfo:
        deglint.GlintCorr(mod_meta_file, product)
    assert f"Error: {band2} does not exist." in str(excinfo)


def test_glintcorr_8():
    """
    Ensure that GlintCorr will raise an Exception if
    fmask information can't be found in odc-metadata.yaml

    This will test GlintCorr.get_fmask_file()
    """
    # load the odc_meta_file
    meta_dict = load_yaml_file(odc_meta_file)

    # change "oa_fmask" key
    meta_dict["measurements"]["no_mask"] = meta_dict["measurements"].pop("oa_fmask")

    # save to data_path where the bands exist. This file will be deleted
    mod_meta_file = data_path / "ga_ls8c_aard_3-2-0_091086_MODIFIED.odc-metadata.yaml"
    with open(mod_meta_file, "w") as ofid:
        yaml.dump(meta_dict, ofid)

    with pytest.raises(Exception) as excinfo:
        deglint.GlintCorr(mod_meta_file, product)
    assert "Unable to extract fmask" in str(excinfo)

    # remove yaml
    mod_meta_file.unlink()
