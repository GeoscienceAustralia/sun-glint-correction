#!/usr/bin/env python3
"""
Unittests for the Hedley et al. (2005) algorithm:
    * Check that deglinted bands are nearly identical to expected
    * Check if vis_im and nir_im have the same dimensions
    * Check if an interative figure is generated when shapefile doesn't exist
    * Check for an empty/faulty shapefile
    * Check if shapefile does not have Polygon geometrie
    * Check that input bands exist
    * Check that Exception is raised for nodata bands
    * Check that correlation plots are created

Notes:
    We will not test if vis_im does not have the
    same dimensions as the nir_im, because
    a resampling occurs if they are not the same.
"""

import pytest
import rasterio
from pathlib import Path
from fiona import collection
from shapely.geometry import Point, LineString, mapping

from sungc import deglint
from . import urd

# specify the path to the odc_metadata.yaml of the test datasets
data_path = Path(__file__).parent / "data"
odc_meta_file = data_path / "ga_ls8c_aard_3-2-0_091086_2014-11-06_final.odc-metadata.yaml"

shp_file = data_path / "HEDLEY" / "ga_ls8c_oa_3-2-0_091086_2014-11-06_final_dw_ROI.shp"

# specify the product
product = "lmbadj"


def test_hedley_image(tmp_path):
    """
    Test if the generated deglinted band is nearly identical
    to expected deglinted band
    """
    g = deglint.GlintCorr(odc_meta_file, product)
    hedley_dir = tmp_path / "HEDLEY"
    hedley_dir.mkdir()

    # ------------------ #
    # Hedley et al. 2005 #
    # ------------------ #
    # deglint the vis bands using band 6
    hedley_xarrlist = g.hedley_2005(
        vis_band_ids=["3"],
        nir_band_id="6",
        roi_shpfile=shp_file,
        overwrite_shp=False,
        odir=hedley_dir,
        water_val=5,
        plot=False,
    )

    sungc_band = hedley_xarrlist[0].lmbadj_green_hedley_deglint.values  # 3D array

    # path to expected sunglint corrected output from Hedley et al.
    exp_sungc_band = (
        data_path
        / "HEDLEY"
        / "ga_ls8c_lmbadj_3-2-0_091086_2014-11-06_final_band03-deglint-600m.tif"
    )


    # ensure that all valid sungint corrected pixels match expected
    with rasterio.open(exp_sungc_band, "r") as exp_sungc_ds:
        urd_band = urd(sungc_band[0, :, :], exp_sungc_ds.read(1), exp_sungc_ds.nodata)
        assert urd_band.max() < 0.001

    # plot=False. Ensure that no png was created in hedley_dir
    png_list = list(hedley_dir.glob("Correlation_*_vs_*.png"))
    assert len(png_list) == 0


def test_hedley_plot(tmp_path):
    """
    Check if a correlation plot (png) is generated. The
    contents of the png are not checked
    """
    hedley_dir = tmp_path / "HEDLEY"
    hedley_dir.mkdir()

    g = deglint.GlintCorr(odc_meta_file, product)
    g.hedley_2005(
        vis_band_ids=["3"],
        nir_band_id="6",
        roi_shpfile=shp_file,
        overwrite_shp=False,
        odir=hedley_dir,
        water_val=5,
        plot=True,  # a plot should be generated in hedley_dir
    )

    # The above function should have generated a png file in
    # hedley_dir with the following naming convention Correlation_*_vs_*.png
    png_list = list(hedley_dir.glob("Correlation_*_vs_*.png"))

    assert len(png_list) == 1


def test_hedley_bands(tmp_path):
    """
    Ensure that hedley_2005() raises an Exception if
    the specifued vis_band_id and nir_band_id do not exist
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    hedley_dir = tmp_path / "HEDLEY"
    hedley_dir.mkdir()

    # check VIS band
    with pytest.raises(Exception) as excinfo:
        g.hedley_2005(
            vis_band_ids=["20"],  # this band doesn't exist
            nir_band_id="6",
            roi_shpfile=shp_file,
            overwrite_shp=False,
            odir=hedley_dir,
            water_val=5,
            plot=False,
        )
    assert "is missing from bands" in str(excinfo)

    # check NIR band
    with pytest.raises(Exception) as excinfo:
        g.hedley_2005(
            vis_band_ids=["3"],
            nir_band_id="20",  # this band doesn't exist
            roi_shpfile=shp_file,
            overwrite_shp=False,
            odir=hedley_dir,
            water_val=5,
            plot=False,
        )
    assert "is missing from bands" in str(excinfo)


def test_empty_band(tmp_path):
    """
    Ensure that hedley_2005() raises an Exception if
    the VIS and NIR band only contain nodata pixels
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    hedley_dir = tmp_path / "HEDLEY"
    hedley_dir.mkdir()

    with pytest.raises(Exception) as excinfo:
        g.hedley_2005(
            vis_band_ids=["7"],  # dummy band only contains nodata (-999)
            nir_band_id="6",
            roi_shpfile=shp_file,
            overwrite_shp=False,
            odir=hedley_dir,
            water_val=5,
            plot=False,
        )
    assert "only contains a single value" in str(excinfo)

    with pytest.raises(Exception) as excinfo:
        g.hedley_2005(
            vis_band_ids=["3"],
            nir_band_id="7",  # dummy band only contains nodata (-999)
            roi_shpfile=shp_file,
            overwrite_shp=False,
            odir=hedley_dir,
            water_val=5,
            plot=False,
        )
    assert "only contains a single value" in str(excinfo)


def test_fake_shp(tmp_path):
    """
    Ensure that hedley_2005() raises an Exception if
    the input shapefile isn't really a shapefile
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    hedley_dir = tmp_path / "HEDLEY"
    hedley_dir.mkdir()

    fake_shp = hedley_dir / "fake.shp"
    with open(fake_shp, "w", encoding="utf-8") as fid:
        fid.write("bogus_geometry")

    with pytest.raises(Exception) as excinfo:
        g.hedley_2005(
            vis_band_ids=["3"],
            nir_band_id="6",
            roi_shpfile=fake_shp,
            overwrite_shp=False,
            odir=hedley_dir,
            water_val=5,
            plot=False,
        )
    assert "not recognized as a supported file format" in str(excinfo)


def test_failure_point_shp(tmp_path):
    """
    Ensure that hedley_2005() raises an Exception if
    the shapefile does not contain any Polygon geometries
    """
    g = deglint.GlintCorr(odc_meta_file, product)

    hedley_dir = tmp_path / "HEDLEY"
    hedley_dir.mkdir()

    # ------------------------------------------- #
    # Test 1: create Points and save to shapefile #
    # ------------------------------------------- #
    point_shp = hedley_dir / "aus_capital_cities.shp"
    # Name, lon (deg. East), lat (deg. North)
    capital_arr = [
        ["PER", 115.8605, -31.9505],
        ["DAR", 130.8456, -12.4634],
        ["ADE", 138.6007, -34.9285],
        ["CAN", 149.1300, -35.2809],
        ["SYD", 151.2093, -33.8688],
        ["BRI", 153.0251, -27.4698],
        ["HOB", 147.3272, -42.8821],
        ["MEL", 144.9631, -37.8136],
    ]
    schema = {"geometry": "Point", "properties": {"name": "str"}}

    with collection(point_shp, "w", "ESRI Shapefile", schema) as shp:
        for cap in capital_arr:
            shp.write(
                {
                    "properties": {"name": cap[0]},
                    "geometry": mapping(Point(cap[1], cap[2])),
                }
            )
    with pytest.raises(Exception) as excinfo:
        g.hedley_2005(
            vis_band_ids=["3"],
            nir_band_id="6",
            roi_shpfile=point_shp,
            overwrite_shp=False,
            odir=hedley_dir,
            water_val=5,
            plot=False,
        )
    assert "input shapefile does not have any 'Polygon' geometry" in str(excinfo)

    # ------------------------------------------------- #
    # Test 2: create a LineString and save to shapefile #
    # ------------------------------------------------- #
    linestr_shp = hedley_dir / "Perth_to_Canberra_line.shp"

    # Name, lon (deg. East), lat (deg. North)
    perth = [115.8605, -31.9505]
    canbr = [149.1300, -35.2809]

    gls = LineString([Point(perth[0], perth[1]), Point(canbr[0], canbr[1])])

    schema = {"geometry": "LineString", "properties": {"name": "str"}}
    with collection(linestr_shp, "w", "ESRI Shapefile", schema) as shp:
        shp.write(
            {
                "properties": {"name": "Perth_to_Canberra"},
                "geometry": mapping(gls),
            }
        )
    with pytest.raises(Exception) as excinfo:
        g.hedley_2005(
            vis_band_ids=["3"],
            nir_band_id="6",
            roi_shpfile=linestr_shp,
            overwrite_shp=False,
            odir=hedley_dir,
            water_val=5,
            plot=False,
        )
    assert "input shapefile does not have any 'Polygon' geometry" in str(excinfo)
