#!/usr/bin/env python3

import os
import yaml
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from datacube.model import Dataset
from rasterio.warp import Resampling

import sungc.rasterio_funcs as rio_funcs
from sungc.interactive import RoiSelector
from sungc.cox_munk_funcs import cm_sunglint
from sungc.visualise import seadas_style_rgb, plot_correlations, display_image

SUPPORTED_SENSORS = [
    "sentinel_2a",
    "sentinel_2b",
    "landsat-5",
    "landsat-7",
    "landsat-8",
    "worldview-2",
]


class GlintCorr:
    def __init__(
        self,
        dc_dataset: Union[Dataset, Path, str],
        product: str,
        fmask_file: Optional[Path] = None,
    ):
        """
        Initiate the sunglint correction class

        Parameters
        ----------
        dc_dataset : datacube.model.Dataset or Path
            Datacube dataset object or *_final.odc-metadata.yaml Path

        product : str
            Product name

        Examples
        --------
        """

        # __init__ uses the metadata dictionary stored in:
        # 1. dc_dataset.metadata_doc, or;
        # 2. *_final.odc-metadata.yaml
        if isinstance(dc_dataset, Dataset):
            metadata_dict = dc_dataset.metadata_doc
            self.group_path = dc_dataset.local_path.parent

        elif isinstance(dc_dataset, Path):
            # check if Path is a yaml... Exception if not
            metadata_dict = self.load_yaml(dc_dataset)
            self.group_path = dc_dataset.parent

        elif isinstance(dc_dataset, str):
            metadata_dict = self.load_yaml(Path(dc_dataset))
            self.group_path = Path(dc_dataset).parent

        else:
            raise ValueError(
                "dc_dataset must be a datacube.model.Dataset, pathlib.Path object or str"
            )

        self.product = product
        self.meas_dict = self.get_meas_dict(metadata_dict)
        self.sensor = self.get_sensor(metadata_dict)

        # self.overpass_datetime isn't used at the moment.
        self.overpass_datetime = self.get_overpass_datetime(metadata_dict)

        # check paths
        self.check_path_exists(self.group_path)

        # check the sensor
        self.check_sensor()

        # define the scale factor to convert to reflectance
        self.scale_factor = 10000.0

        # get list of tif files from self.meas_dict
        self.band_list, self.band_ids, self.bandnames = self.get_band_list()

        # get a useful output basename
        self.useful_obase = self.output_basename(self.band_list[0])

        # get a useful deep-water ROI shapefil
        self.obase_shp = self.useful_obase + "_deepWater_ROI.shp"

        if fmask_file:
            self.fmask_file = Path(fmask_file)
        else:
            # get fmask from self.meas_dict
            self.fmask_file = self.get_fmask_file()

        self.check_path_exists(self.fmask_file)

    def load_yaml(self, p: Path):
        if ".odc-metadata.yaml" not in p.name:
            raise Exception(f"{p} is not a .odc-metadata.yaml")

        with p.open() as f:
            contents = yaml.load(f, Loader=yaml.FullLoader)
        return contents

    def output_basename(self, filename: Path):
        """ Get a useful output basename """
        return filename.stem.split("_final")[0]

    def deglint_ofile(
        self, spatial_res: Union[float, int], out_dir: Path, visible_bfile: Path
    ):
        """ Get deglint filename """
        vis_bname = visible_bfile.stem
        return out_dir / "{0}-deglint-{1}m.tif".format(vis_bname, spatial_res)

    def check_sensor(self):
        if self.sensor not in SUPPORTED_SENSORS:
            msg = f"Supported sensors are: {SUPPORTED_SENSORS}, recieved {self.sensor}"
            raise Exception(msg)

    def check_path_exists(self, path: Path):
        """ Checks if a path exists """
        if not path.exists():
            raise Exception("\nError: {0} does not exist.".format(path))

    def check_bandnum_exist(self, bandnums: list, required_bandnums: list):
        """
        Checks if the band numbers in the required list exist

        Parameters
        ----------
        bandnums : list
            A list of band numbers

        required_bandnums : list
            A list of required band numbers

        Raises
        ------
        Exception
            if any required band numbers are missing
            from bandnums
        """
        for req_bn in required_bandnums:
            if not (req_bn in bandnums):
                raise Exception(
                    "B{0} is missing from bands [{1}]".format(
                        req_bn, ", ".join(["B" + str(i) for i in bandnums])
                    )
                )

    def find_file(self, keyword):
        """
        find a file in a directory (self.meas_dict)
        has a keyword in its filename

        Parameters
        ----------
        keyword : str
            keyword

        Returns
        -------
        filename : str
            filename

        Raises
        ------
        Exception if filename not found
        """
        filename = None
        for key in self.meas_dict.keys():
            if key.find(keyword) != -1:
                basename = self.meas_dict[key]["path"]
                filename = self.group_path.joinpath(basename)

        if not filename:
            raise Exception("\nfilename with keyword ({0}) not found".format(keyword))

        return filename

    def get_meas_dict(self, metadata_dict):
        """
        Get the measurement dictionary from the datacube dataset.

        Parameters
        ----------
        metadata_dict : dict
            metadata dictionary

        Returns
        -------
        meas_dict : dict
            A dictionary containing band information
        """
        if "image" in metadata_dict:
            # older version
            try:
                meas_dict = metadata_dict["image"]["bands"]
            except KeyError:
                raise Exception(
                    'unable to extract bands, "bands" not in metadata_dict["image"]'
                )

        elif "measurements" in metadata_dict:
            # newer version
            meas_dict = metadata_dict["measurements"]
        else:
            raise Exception(
                "neither image nor measurements keys were found in metadata_dict"
            )

        return meas_dict

    def get_sensor(self, metadata_dict):
        """
        Get the sensor from the datacibe dataset

        Parameters
        ----------
        metadata_dict : dict
            metadata dictionary

        Returns
        -------
        sensor : str
            The sensor name
        """
        msg = "Unable to extract sensor name"
        if "platform" in metadata_dict:
            # older version
            try:
                sensor = metadata_dict["platform"]["code"].lower()
            except KeyError:
                raise Exception(f'{msg}, "code" not in metadata_dict["platform"]')

        elif "properties" in metadata_dict:
            # newer version
            try:
                sensor = metadata_dict["properties"]["eo:platform"].lower()
            except KeyError:
                raise Exception(
                    f'{msg}, "eo:platform" not in metadata_dict["properties"]'
                )

        else:
            raise Exception(
                "neither platform nor properties keys were found in metadata_dict"
            )

        return sensor

    def get_overpass_datetime(self, metadata_dict):
        """
        Get the overpass datetime from the datacibe dataset

        Parameters
        ----------
        metadata_dict : dict
            metadata dictionary

        Returns
        -------
        overpass_datetime : datetime
            YYYY-MM-DD HH:MM:SS.SSSSSS
        """
        overpass_datetime = None
        msg = "Unable to extract overpass datetime"
        if "extent" in metadata_dict:
            # new and old metadata have "extent" key
            if "center_dt" in metadata_dict["extent"]:
                # older version
                overpass_datetime = datetime.strptime(
                    metadata_dict["extent"]["center_dt"],
                    "%Y-%m-%dT%H:%M:%S.%fZ",
                )
            else:
                msg += ', "center_dt" not in metadata_dict["extent"]'

        if "properties" in metadata_dict:
            # newer version - assume a datetime object
            if "datetime" in metadata_dict["properties"]:
                overpass_datetime = metadata_dict["properties"]["datetime"]

                if isinstance(overpass_datetime, str):
                    overpass_datetime = datetime.strptime(
                        metadata_dict["properties"]["datetime"], "%Y-%m-%d %H:%M:%S.%fZ"
                    )
            else:
                msg += ', "datetime" not in metadata_dict["properties"]'

        if not overpass_datetime:
            if msg == "Unable to extract overpass datetime":
                msg += '"extent" or "properties" not in metadata_dict'

            raise Exception(f"{msg}")

        return overpass_datetime

    def get_band_list(self):
        """
        Get a list of tifs in from self.meas_dict

        Returns
        -------
        bandlist : list
            A list of pathlib

        bandnums : list of strings
            A list of band numbers

        bandnames : list of string
            A list of band names

        Raises:
            Exception if any of the bands do not exist
        """
        # a list of bands that are not needed for deglinting
        skip_bands = [
            "contiguity",
            "azimuthal-exiting",
            "azimuthal-incident",
            "combined-terrain-shadow",
            "relative-slope",
            "satellite-azimuth",
            "solar-azimuth",
            "time-delta",
            "exiting-angle",
            "incident-angle",
        ]

        bandlist = []
        bandnums = []
        bandnames = []
        for key in self.meas_dict.keys():
            key_spl = key.split("_")
            if key_spl[0].find(self.product) != -1:
                basename = self.meas_dict[key]["path"]

                # skip over files that aren't needed for deglinting
                if basename.lower() in skip_bands:
                    continue

                bfile = self.group_path.joinpath(basename)

                # ensure that bfile exists, raise exception if not
                self.check_path_exists(bfile)

                bnum = os.path.splitext(basename)[0][-2:].strip("0")

                bandnames.append("_".join(key_spl[1:]))
                bandnums.append(bnum)
                bandlist.append(bfile)

        if not bandlist:
            raise Exception("Could not find any geotifs in '{0}'".format(self.group_path))

        return bandlist, bandnums, bandnames

    def get_fmask_file(self):
        """
        Get the fmask file from self.meas_dict

        Returns
        -------
        fmask_file : str
            fmask filename

        Raises
        ------
        Exception if fmask file is not found
        """
        msg = "Unable to extract fmask"
        if "fmask" in self.meas_dict:
            if "path" in self.meas_dict["fmask"]:
                fmask_file = self.group_path.joinpath(self.meas_dict["fmask"]["path"])
            else:
                raise Exception(f'{msg}, "path" not in self.meas_dict["fmask"]')

        elif "oa_fmask" in self.meas_dict:
            if "path" in self.meas_dict["oa_fmask"]:
                fmask_file = self.group_path.joinpath(self.meas_dict["oa_fmask"]["path"])
            else:
                raise Exception(f'{msg}, "path" not in self.meas_dict["oa_fmask"]')

        else:
            raise Exception(f'{msg}, "fmask" or "oa_fmask" not in self.meas_dict')

        return fmask_file

    def quicklook_rgb(self, dwnscale_factor: float = 3):
        """
        Generate a quicklook from the sensor's RGB bands

        Parameters
        ----------
        dwnscale_factor : float >= 1
            The downscaling factor.
            If dwnscale_factor = 3 then the spatial resolution
            of the quicklook RGB will be reduced by a factor
            of three from the native resolution of the sensors'
            RGB bands.

            If dwnscale_factor = 1 then no downscaling is
            performed, thus the native resolution of the RGB
            bands are used.

            e.g. if dwnscale_factor = 3, then the 10 m RGB bands
            will be downscaled to 30 m spatial resolution

        Returns
        -------
        rgb_im : numpy.ndarray
            RGB image with the following dimensions
            [nrows, ncols, 3] for the three channels

        rgb_meta : dict
            Metadata dictionary taken from rasterio

        Raises
        ------
        Exception
            * if dwnscale_factor < 1
        """
        if dwnscale_factor < 1:
            raise Exception("\ndwnscale_factor must be a float >= 1")

        if (
            (self.sensor == "sentinel_2a")
            or (self.sensor == "sentinel_2b")
            or (self.sensor == "landsat-8")
        ):
            ix_red = self.band_ids.index("4")
            ix_grn = self.band_ids.index("3")
            ix_blu = self.band_ids.index("2")

        if (self.sensor == "landsat-5") or (self.sensor == "landsat-7"):
            ix_red = self.band_ids.index("3")
            ix_grn = self.band_ids.index("2")
            ix_blu = self.band_ids.index("1")

        if self.sensor == "worldview-2":
            ix_red = self.band_ids.index("5")
            ix_grn = self.band_ids.index("3")
            ix_blu = self.band_ids.index("2")

        rgb_bandlist = [
            self.band_list[ix_red],
            self.band_list[ix_grn],
            self.band_list[ix_blu],
        ]
        with rasterio.open(rgb_bandlist[0], "r") as ds:
            ql_spatial_res = dwnscale_factor * float(ds.transform.a)

        if dwnscale_factor > 1:
            # resample to quicklook spatial resolution
            resmpl_tifs, refl_im, rio_meta = rio_funcs.resample_bands(
                rgb_bandlist,
                ql_spatial_res,
                Resampling.nearest,
                load=True,
                save=False,
                odir=None,
            )
        else:
            refl_im, rio_meta = rio_funcs.load_bands(
                rgb_bandlist, self.scale_factor, False
            )

        # use NASA-OBPG SeaDAS's transformation to create
        # a very pretty RGB
        rgb_im = seadas_style_rgb(
            refl_img=refl_im, rgb_ix=[0, 1, 2], scale_factor=self.scale_factor
        )

        rio_meta["band_1"] = rgb_bandlist[0].stem
        rio_meta["band_2"] = rgb_bandlist[1].stem
        rio_meta["band_3"] = rgb_bandlist[2].stem
        rio_meta["dtype"] = rgb_im.dtype.name  # this should np.unit8

        return rgb_im, rio_meta

    def create_roi_shp(self, shp_file, dwnscaling_factor=3):
        """
        Create a shapefile containing a polygon of
        a ROI that is selected using the interactive
        quicklook RGB

        Parameters
        ----------
        shp_file : str
            shapefile containing a polygon

        dwnscale_factor : float >= 1
            The downscaling factor used to downscale the native
            RGB bands to generate the quicklook RGB.

            If dwnscale_factor = 3 then the spatial resolution
            of the quicklook RGB will be reduced by a factor
            of three from the native resolution of the sensors'
            RGB bands.

            If dwnscale_factor = 1 then no downscaling is
            performed, thus the native resolution of the RGB
            bands are used.

            e.g. if dwnscale_factor = 3, then the 10 m RGB bands
            will be downscaled to 30 m spatial resolution

        Rasies:
            Exception if dwnscaling_factor < 1
        """
        # generate a downscaled quicklook
        rgb_im, rgb_meta = self.quicklook_rgb(dwnscaling_factor)

        # let the user select a ROI from the quicklook RGB
        ax = display_image(rgb_im, None)
        mc = RoiSelector(ax=ax)
        mc.interative()
        plt.show()

        # write a shapefile
        mc.verts_to_shp(metadata=rgb_meta, shp_file=shp_file)

        # close the RoiSelector
        mc = None

    def nir_subtraction(self, vis_band_ids, nir_band_id, odir=None, water_val=5):
        """
        This sunglint correction assumes that glint reflectance
        is nearly spectrally flat in the VIS-NIR. Hence, the NIR
        reflectance is subtracted from the VIS bands.

        Dierssen, H.M., Chlus, A., Russell, B. 2015. Hyperspectral
        discrimination of floating mats of seagrass wrack and the
        macroalgae Sargassum in coastal waters of Greater Florida
        Bay using airborne remote sensing. Remote Sens. Environ.,
        167(15), 247-258, doi: https://doi.org/10.1016/j.rse.2015.01.027


        Parameters
        ----------
        vis_band_ids : list
            A list of band numbers in the visible that will be deglinted

        nir_band_id : str
            The NIR band number used to deglint the VIS bands in vis_band_ids

        odir : str
            The path where the deglinted geotiff bands are saved.

            if None then:
            odir = self.group_path / "DEGLINT" / "MINUS_NIR"

        water_val : int
            The fmask value for water pixels (default = 5)

        Returns
        -------
        deglint_bandlist : list
            A list of paths to the deglinted geotiff bands

        Notes
        -----
        1) fmask file is resampled to spatial resolution
           to the VIS bands using Rasterio
           (Resampling.bilinear)

        Raises
        ------
        Exception
            * If any of the bands do not exist
            * If any of the input data only contains nodata
        """
        # create output directory
        if not odir:
            odir = self.group_path / "DEGLINT" / "MINUS_NIR"

        if not odir.exists:
            odir.mkdir(exist_ok=True)

        # Check that the input vis bands exist
        self.check_bandnum_exist(self.band_ids, vis_band_ids)
        self.check_bandnum_exist(self.band_ids, [nir_band_id])

        ix_nir = self.band_ids.index(nir_band_id)
        nir_bandpath = self.band_list[ix_nir]  # Path

        # ------------------------------ #
        #       load the NIR band        #
        # ------------------------------ #
        with rasterio.open(nir_bandpath, "r") as nir_ds:
            nir_nrows = nir_ds.height
            nir_ncols = nir_ds.width
            nir_im = nir_ds.read(1)

            if nir_ds.nodata is not None:
                rio_funcs.check_image_singleval(nir_im, nir_ds.nodata, "nir_im")

        # ------------------------------ #
        nbands = len(vis_band_ids)
        deglint_bandlist = []

        # Iterate over each visible band
        for z in range(0, nbands):

            ix_vis = self.band_ids.index(vis_band_ids[z])
            vis_bandpath = self.band_list[ix_vis]  # Path

            # ------------------------------ #
            #        load visible band       #
            # ------------------------------ #
            with rasterio.open(vis_bandpath, "r") as ds_vis:
                vis_im = ds_vis.read(1)

                # get metadata and load NIR array
                kwargs = ds_vis.meta.copy()
                nodata = kwargs["nodata"]

                if nodata is not None:
                    rio_funcs.check_image_singleval(vis_im, nodata, "vis_im")

                spatial_res = int(abs(kwargs["transform"].a))

                # ------------------------------ #
                #  Resample and load fmask_file  #
                # ------------------------------ #
                fmask = rio_funcs.resample_file_to_ds(
                    self.fmask_file, ds_vis, Resampling.mode
                )

                # ------------------------------ #
                #        Resample NIR band       #
                #               and              #
                #       Sunglint Correction      #
                # ------------------------------ #
                deglint_band = np.array(vis_im, order="K", copy=True)

                if (nir_nrows != ds_vis.height) or (nir_ncols != ds_vis.width):
                    # resample the NIR band to match the VIS band
                    nir_im_res = rio_funcs.resample_file_to_ds(
                        nir_bandpath, ds_vis, Resampling.bilinear
                    )

                    water_ix = np.where(
                        (fmask == water_val) & (vis_im != nodata) & (nir_im_res != nodata)
                    )

                    # 1. deglint water pixels
                    deglint_band[water_ix] = vis_im[water_ix] - nir_im_res[water_ix]

                    # Apply mask:
                    deglint_band[(vis_im == nodata) | (nir_im_res == nodata)] = nodata

                else:
                    water_ix = np.where(
                        (fmask == water_val) & (vis_im != nodata) & (nir_im != nodata)
                    )

                    deglint_band[water_ix] = vis_im[water_ix] - nir_im[water_ix]
                    deglint_band[(vis_im == nodata) | (nir_im == nodata)] = nodata

            # 2. write geotiff
            deglint_otif = self.deglint_ofile(spatial_res, odir, vis_bandpath)
            with rasterio.open(deglint_otif, "w", **kwargs) as dst:
                dst.write(deglint_band, 1)

            if deglint_otif.exists():
                deglint_bandlist.append(deglint_otif)

        # endfor z
        return deglint_bandlist

    def hedley_2005(
        self,
        vis_band_ids,
        nir_band_id,
        roi_shpfile=None,
        overwrite_shp=False,
        odir=None,
        water_val=5,
        plot=False,
    ):
        """
        Sunglint correction using the algorithm:
        Hedley, J. D., Harborne, A. R., Mumby, P. J. (2005). Simple and
        robust removal of sun glint for mapping shallow-water benthos.
        International Journal of Remote Sensing, 26(10), 2107-2112.

        Parameters
        ----------
        vis_band_ids : list
            A list of band numbers in the visible that will be deglinted

        nir_band_id : str
            The NIR band number used to deglint the VIS bands in vis_band_ids

        roi_shpfile : str
            Path to shapefile containing a polygon of a deep
            water region containing a range of sunglint
            contaminated pixels

        overwrite_shp : bool (True | False)
            Overwrite the shapefile containing a polygon
            of the sunglint contaminated deep-water region.
            True  -> overwrites a shapefile (if it exists)
            False -> uses specified shapefile

        odir : str
            The path where the deglinted geotiff bands
            and correlation plots (if specified) are saved.

            if None then:
            odir = self.group_path / "DEGLINT" / "HEDLEY"

        water_val : int
            The fmask value for water pixels (default = 5)

        plot : bool (True | False)
            True will save the correlation plots to the odir specified above.

        Returns
        -------
        deglint_bandlist : list
            A list of paths to the deglinted geotiff bands

        Notes
        -----
        1) fmask file is resampled to spatial resolution
           to the VIS bands using Rasterio
           (Resampling.bilinear)

        2) Python's rasterio and fiona are used to load
           the shapefile as a mask using the NIR band
           as the reference.

        3) Individual correlation plots are generated
           between each VIS band and the NIR band.

        4) nir band is upscaled/downscaled to the
           vis bands.

        Raises
        ------
        Exception
            * If any of the bands do not exist
        """
        # get name of the sunglint contaminated
        # deep water region polygon ascii file
        if not roi_shpfile:
            roi_shpfile = self.group_path / "deepWater_ROI_polygon.shp"

        if not roi_shpfile.exists():
            # use interactive mode to generate the shapefile
            self.create_roi_shp(roi_shpfile)
        else:
            if overwrite_shp:
                # recreate shapefile
                self.create_roi_shp(roi_shpfile)

        # create output directory
        if not odir:
            odir = self.group_path / "DEGLINT" / "HEDLEY"

        if not odir.exists():
            odir.mkdir(exist_ok=True)

        # initiate plot if specified
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # Check that the input vis bands exist
        self.check_bandnum_exist(self.band_ids, vis_band_ids)
        self.check_bandnum_exist(self.band_ids, [nir_band_id])

        ix_nir = self.band_ids.index(nir_band_id)
        nir_bandpath = self.band_list[ix_nir]  # Path

        # ------------------------------ #
        #       load the NIR band        #
        # ------------------------------ #
        with rasterio.open(nir_bandpath, "r") as nir_ds:
            nir_nrows = nir_ds.height
            nir_ncols = nir_ds.width
            nir_im_orig = nir_ds.read(1)

            if nir_ds.nodata is not None:
                rio_funcs.check_image_singleval(
                    nir_im_orig, nir_ds.nodata, "nir_im_orig"
                )

        # ------------------------------ #
        nbands = len(vis_band_ids)
        deglint_bandlist = []

        # Iterate over each visible band
        for z in range(0, nbands):

            ix_vis = self.band_ids.index(vis_band_ids[z])
            vis_bandpath = self.band_list[ix_vis]  # Path

            # ------------------------------ #
            #        load visible band       #
            # ------------------------------ #
            with rasterio.open(vis_bandpath, "r") as ds_vis:
                vis_im = ds_vis.read(1)

                # get metadata and load NIR array
                kwargs = ds_vis.meta.copy()
                nodata = kwargs["nodata"]
                spatial_res = int(abs(kwargs["transform"].a))

                if nodata is not None:
                    rio_funcs.check_image_singleval(vis_im, nodata, "vis_im")

                # ------------------------------ #
                #       Resample NIR band        #
                # ------------------------------ #
                if (nir_nrows != ds_vis.height) or (nir_ncols != ds_vis.width):
                    # resample the NIR band to match the VIS band
                    nir_im = rio_funcs.resample_file_to_ds(
                        nir_bandpath, ds_vis, Resampling.bilinear
                    )

                else:
                    nir_im = np.copy(nir_im_orig)

                # ------------------------------ #
                #  Resample and load fmask_file  #
                # ------------------------------ #
                fmask = rio_funcs.resample_file_to_ds(
                    self.fmask_file, ds_vis, Resampling.mode
                )

                # ------------------------------ #
                #    Load shapefile as a mask    #
                # ------------------------------ #
                roi_mask = rio_funcs.load_mask_from_shp(roi_shpfile, ds_vis)

            flag_mask = (fmask != water_val) | (vis_im == nodata) | (nir_im == nodata)
            water_mask = ~flag_mask
            roi_mask[flag_mask] = nodata

            # ------------------------------ #
            #       Sunglint Correction      #
            # ------------------------------ #
            # copy band
            deglint_band = np.array(vis_im, order="K", copy=True)

            # 1. Find minimum NIR in the roi polygon
            roi_valix = np.where(roi_mask != nodata)
            valid_nir = nir_im[roi_valix]
            min_refl_nir = valid_nir.min()

            # 2. Get correlations between current band and NIR
            y_vals = vis_im[roi_valix]
            slope, y_inter, r_val, p_val, std_err = stats.linregress(
                x=valid_nir, y=y_vals
            )

            # 3. deglint water pixels
            deglint_band[water_mask] = vis_im[water_mask] - slope * (
                nir_im[water_mask] - min_refl_nir
            )
            deglint_band[(vis_im == nodata) | (nir_im == nodata)] = nodata

            # 4. write geotiff
            deglint_otif = self.deglint_ofile(spatial_res, odir, vis_bandpath)
            with rasterio.open(deglint_otif, "w", **kwargs) as dst:
                dst.write(deglint_band, 1)

            if deglint_otif.exists():
                deglint_bandlist.append(deglint_otif)

            # ------------------------------ #
            #        Plot correlations       #
            # ------------------------------ #
            if plot:
                # create a density plot
                plot_correlations(
                    fig=fig,
                    ax=ax,
                    r2=r_val ** 2,
                    slope=slope,
                    y_inter=y_inter,
                    nir_vals=valid_nir,
                    vis_vals=y_vals,
                    scale_factor=self.scale_factor,
                    nir_band_id=nir_band_id,
                    vis_band_id=vis_band_ids[z],
                    odir=odir,
                )
        # endfor z

        return deglint_bandlist

    def cox_munk(
        self,
        vis_band_ids: list,
        odir: Union[Path, None] = None,
        vzen_file: Union[Path, None] = None,
        szen_file: Union[Path, None] = None,
        razi_file: Union[Path, None] = None,
        wind_speed: float = 5,
        water_val: int = 5,
    ):
        """
        Performs the wind-direction-independent Cox and Munk (1954)
        sunglint correction on the specified visible bands. This
        algorithm is suitable for spatial resolutions between
        100 - 1000 metres (Kay et al., 2009). Fresnel reflectance
        of sunglint is assumed to be wavelength-independent.

        Cox, C., Munk, W. 1954. Statistics of the Sea Surface Derived
        from Sun Glitter. J. Mar. Res., 13, 198-227.

        Cox, C., Munk, W. 1954. Measurement of the Roughness of the Sea
        Surface from Photographs of the Suns Glitter. J. Opt. Soc. Am.,
        44, 838-850.

        Kay, S., Hedley, J. D., Lavender, S. 2009. Sun Glint Correction
        of High and Low Spatial Resolution Images of Aquatic Scenes:
        a Review of Methods for Visible and Near-Infrared Wavelengths.
        Remote Sensing, 1, 697-730; doi:10.3390/rs1040697

        Parameters
        ----------
        vis_band_ids : list
            A list of band numbers in the visible that will be deglinted

        odir : Path or None
            The path where the deglinted geotiff bands are saved.

            if None then:
            odir = self.group_path / "DEGLINT" / "COX_MUNK"

        vzen_file : Path or None
            sensor view zenith (rasterio-openable) image file

            * if None then the file containing "satellite-view.tif"
              inside group_path is designated as the vzen_file

        szen_file : Path or None
            solar zenith (rasterio-openable) image file

            * if None then the file containing "solar-zenith.tif"
              inside group_path is designated as the szen_file

        razi_file : Path or None
            Relative azimuth (rasterio-openable) image file
            Relative azimuth = solar_azimuth - sensor_azimuth

            * if None then the file containing "relative-azimuth.tif"
              inside group_path is designated as the razi_file

        wind_speed : float
            Wind speed (m/s)

        water_val : int
            The fmask value for water pixels (default = 5)

        Returns
        -------
        deglint_bandlist : list
            A list of paths to the deglinted geotiff bands

        Raises
        ------
        Exception:
            * if input arrays are not two-dimensional
            * if any input arrays only contain nodata
            * if dimension mismatch
            * if wind_speed < 0
        """
        # create output directory
        if not odir:
            odir = self.group_path / "DEGLINT" / "COX_MUNK"

        if not odir.exists():
            os.makedirs(odir, exist_ok=True)

        # --- check vzen_file --- #
        if vzen_file:
            self.check_path_exists(vzen_file)
        else:
            # find view zenith from self.meas_dict
            vzen_file = self.find_file("satellite_view")

        # --- check szen_file --- #
        if szen_file:
            self.check_path_exists(szen_file)
        else:
            # find solar zenith from self.meas_dict
            szen_file = self.find_file("solar_zenith")

        # --- check razi_file --- #
        if razi_file:
            self.check_path_exists(razi_file)
        else:
            # find relative azimuth from self.meas_dict
            razi_file = self.find_file("relative_azimuth")

        # Check that the input vis bands exist
        self.check_bandnum_exist(self.band_ids, vis_band_ids)

        # ------------------------------- #
        #  Estimate sunglint reflectance  #
        # ------------------------------- #

        # cox and munk:
        p_glint, p_fresnel, cm_meta = cm_sunglint(
            view_zenith_file=vzen_file,
            solar_zenith_file=szen_file,
            relative_azimuth_file=razi_file,
            wind_speed=wind_speed,
            return_fresnel=False,
        )
        # although the fresnel reflectance (p_fresnel) is not used,
        # it is useful to keep for testing/debugging

        p_nodata = cm_meta["nodata"]  # this is np.nan

        psg_nrows, psg_ncols = p_glint.shape
        # Notes:
        #    * if return_fresnel=True  then p_fresnel = numpy.ndarray
        #    * if return_fresnel=False then p_fresnel = None
        #    * 0 <= p_glint   (np.float32) <= 1
        #    * 0 <= p_fresnel (np.float32) <= 1

        # In this implementation, the scale_factor (usually 10,000)
        # will not be applied to minimise storage of deglinted
        # bands. Hence, we need to multiply p_glint by this factor
        p_glint[p_glint != p_nodata] *= self.scale_factor  # keep as np.float32

        # ------------------------------ #
        nbands = len(vis_band_ids)
        deglint_bandlist = []

        for z in range(0, nbands):

            ix_vis = self.band_ids.index(vis_band_ids[z])
            vis_bandpath = self.band_list[ix_vis]  # Path

            # ------------------------------ #
            #        load visible band       #
            # ------------------------------ #
            with rasterio.open(vis_bandpath, "r") as ds_vis:
                vis_im = ds_vis.read(1)
                vis_nrows, vis_ncols = vis_im.shape

                # get metadata
                kwargs = ds_vis.meta.copy()
                nodata = kwargs["nodata"]  # this is -999
                spatial_res = int(abs(kwargs["transform"].a))

                if nodata is not None:
                    rio_funcs.check_image_singleval(vis_im, nodata, "vis_im")

                # do this once!
                if z == 0:
                    if p_nodata != nodata:
                        p_glint[p_glint == p_nodata] = nodata
                        if isinstance(p_fresnel, np.ndarray):
                            p_fresnel[p_fresnel != p_nodata] *= self.scale_factor
                            p_fresnel[p_fresnel == p_nodata] = nodata
                            # convert from np.float32 to np.int16
                            p_fresnel = np.array(p_fresnel, order="C", dtype=vis_im.dtype)

                # ------------------------------ #
                #  Resample and load fmask_file  #
                # ------------------------------ #
                fmask = rio_funcs.resample_file_to_ds(
                    self.fmask_file, ds_vis, Resampling.mode
                )

                # ------------------------------ #
                #        Resample p_glint        #
                # ------------------------------ #
                if (vis_nrows != psg_nrows) or (psg_ncols != vis_ncols):
                    p_glint_res = rio_funcs.resample_band_to_ds(
                        p_glint, cm_meta, ds_vis, Resampling.bilinear
                    )

                else:
                    p_glint_res = np.copy(p_glint)

            # ------------------------------ #
            #       Sunglint Correction      #
            # ------------------------------ #
            # copy band
            deglint_band = np.array(vis_im, order="K", copy=True)
            water_msk = (fmask == water_val) & (vis_im != nodata)

            # 1. deglint water pixels
            deglint_band[water_msk] = vis_im[water_msk] - p_glint_res[water_msk]

            # 2. write geotiff
            deglint_otif = self.deglint_ofile(spatial_res, odir, vis_bandpath)
            with rasterio.open(deglint_otif, "w", **kwargs) as dst:
                dst.write(deglint_band, 1)

            if deglint_otif.exists():
                deglint_bandlist.append(deglint_otif)

        # endfor z
        return deglint_bandlist
