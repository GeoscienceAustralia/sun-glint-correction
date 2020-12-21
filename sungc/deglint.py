#!/usr/bin/env python3

import os
import yaml
import tempfile
import rasterio
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from datacube.model import Dataset
from rasterio.warp import Resampling
from typing import Optional, Union, List, Tuple

import sungc.rasterio_funcs as rf
from sungc.xarr_funcs import create_xr
from sungc.algorithms import subtract_backend, coxmunk_backend, hedley_backend

SUPPORTED_SENSORS = [
    "sentinel2a",
    "sentinel2b",
    "landsat5",
    "landsat7",
    "landsat8",
    "worldview2",
]


class GlintCorrX:
    """
    xarray version of GlintCorr
    """

    def __init__(
        self,
        xarr: xr.Dataset,
        scale_factor: Union[int, float] = 10000.0,
    ):
        """
        Initiate the sunglint correction class for xarray inputs

        Parameters
        ----------
        xarr: xr.Dataset
            xarray object containing resampled bands all at the
            same spatial resolution.

        scale_factor: int or float
            factor used to convert int16 to float32 (default = 10000)
        """
        if not isinstance(xarr, xr.Dataset):
            raise ValueError("input xarr must be xarray.Dataset")

        self.xarr = xarr
        self.scale_factor = scale_factor
        self.ntime = self.xarr.dims["time"]
        self.nrows = self.xarr.dims["y"]
        self.ncols = self.xarr.dims["x"]

    def check_bands_in_xarr(self, input_bands: List[str]):
        """
        Check if all of the data variable names in input_bands
        exist in the xarray. If any of the variable names do
        not exist then a ValueError is raised.
        """
        if not all([varname in self.xarr.data_vars for varname in input_bands]):
            raise ValueError(
                f"One or more variable data names ({input_bands}) "
                "specified do not exist in the xarray"
            )

    def check_roi_shplist(self, roi_shplist: Union[List[Path], List[str]]):
        """ Check roi_shplist """
        # check if numb in list = ntime
        if len(roi_shplist) != self.ntime:
            raise ValueError(
                f"number of shapefiles in list ({len(roi_shplist)}) "
                f"!= number of timestamps ({self.ntime})"
            )

        # ensure roi_shplist contains the correct input type
        shp_type_check = [isinstance(shp_, (Path, str)) for shp_ in roi_shplist]
        if not all(shp_type_check):
            raise ValueError(
                "roi_shplist contains elements that are not pathlib.Path or str objects."
            )

        # ensure all shapefiles in roi_shplist exist
        roi_exist_check = [Path(shp_).exists() for shp_ in roi_shplist]
        if not all(roi_exist_check):
            raise ValueError("roi_shplist contains shapefiles that do not exist.")

    def glint_subtraction(
        self,
        vis_bands: List[str],
        corr_band: str,
    ) -> xr.Dataset:
        """
        This sunglint correction assumes that glint reflectance
        is nearly spectrally flat in the VIS-NIR. Hence, the NIR
        or SWIR reflectance is subtracted from the VIS bands.

        Dierssen, H.M., Chlus, A., Russell, B. 2015. Hyperspectral
        discrimination of floating mats of seagrass wrack and the
        macroalgae Sargassum in coastal waters of Greater Florida
        Bay using airborne remote sensing. Remote Sens. Environ.,
        167(15), 247-258, doi: https://doi.org/10.1016/j.rse.2015.01.027


        Parameters
        ----------
        vis_bands : list
            A list of data variable names present in the xarray for
            the visible bands that will be deglinted.

        corr_band : str
            The variable name in the xarray for the NIR/SWIR band
            that will be used to deglint the VIS bands.

        Returns
        -------
        deglinted_dxr : xarray.Dataset
            xarray object containing the deglinted vis_bands

        Raises
        ------
        ValueError
            * If any of the input variable names do not exist

        Notes
        -----
        It is assumed that all bands in the xarray have the
        spatial resolution and that non-water pixels have
        already been masked.
        """
        # first we need to check if the variable names in
        # vis_bands and corr_band exist in the xarray
        self.check_bands_in_xarr(vis_bands)
        self.check_bands_in_xarr([corr_band])

        # get the nir/swir band
        corr_im = self.xarr.variables[corr_band].values  # np.ndarray

        xr_dvars = {}

        for visname in vis_bands:
            vis_im = self.xarr.variables[visname].values
            nodata = self.xarr.variables[visname].attrs["nodata"]
            dtype = self.xarr.variables[visname].dtype  # <class 'numpy.dtype'>

            deglinted_im = np.zeros(
                [self.ntime, self.nrows, self.ncols], order="C", dtype=dtype
            )

            success_bands = []
            # iterate through time and deglint
            for t in range(self.ntime):
                water_mask = (vis_im[t, :, :] != nodata) & (corr_im[t, :, :] != nodata)

                # check for nodata/empty arrays:
                if rf.check_singleval(vis_im[t, :, :], nodata) or rf.check_singleval(
                    corr_im[t, :, :], nodata
                ):
                    deglinted_im[t, :, :] = np.full((self.nrows, self.ncols), nodata)
                    success_bands.append(False)
                    continue  # go to next timestamp

                deglinted_im[t, :, :] = subtract_backend(
                    vis_band=vis_im[t, :, :],
                    corr_band=corr_im[t, :, :],
                    water_mask=water_mask,
                    nodata=nodata,
                    scale_factor=self.scale_factor,
                    clip=False,
                )
                success_bands.append(True)

            xr_dvars[f"{visname}"] = xr.Variable(
                dims=["time", "y", "x"],
                data=deglinted_im,
                attrs={"deglint_success": success_bands},
            )

        # create the output xarray.Dataset
        new_attrs = self.xarr.attrs.copy()
        new_attrs["deglint_algorithm"] = "NIR/SWIR subtraction"

        new_coords = {
            "time": self.xarr.coords["time"].values,
            "y": self.xarr.coords["y"].values,
            "x": self.xarr.coords["x"].values,
        }

        deglinted_dxr = xr.Dataset(data_vars=xr_dvars, coords=new_coords, attrs=new_attrs)

        return deglinted_dxr

    def hedley_2005(
        self,
        vis_bands: List[str],
        corr_band: str,
        plot: bool = False,
        roi_shplist: Optional[Union[List[Path], List[str]]] = None,
        rgb_varlist: Optional[List[str]] = None,
        dwnscale_factor: Optional[float] = 3.0,
        odir: Optional[Path] = None,
    ) -> xr.Dataset:
        """
        Sunglint correction using the algorithm:
        Hedley, J. D., Harborne, A. R., Mumby, P. J. (2005). Simple and
        robust removal of sun glint for mapping shallow-water benthos.
        International Journal of Remote Sensing, 26(10), 2107-2112.

        Parameters
        ----------
        vis_bands : list
            A list of data variable names present in the xarray for
            the visible bands that will be deglinted.

        corr_band : str
            The variable name in the xarray for the NIR/SWIR band
            that will be used to deglint the VIS bands.

        plot : bool (True | False)
            True will save the correlation plots to the odir specified above.

        roi_shplist : List or None
            A list of Paths to the shapefile for each timestamp in the
            xarray. The shapefiles should have a polygon of a deep
            water region containing a range of sunglint contaminated pixels

        dwnscale_factor : float >= 1
            The downscaling factor used to downscale the native RGB
            bands to generate the quicklook RGB.

            If dwnscale_factor = 3 then the spatial resolution of the
            quicklook RGB will be reduced by a factor of three

            If dwnscale_factor = 1 then no downscaling is performed

        odir : Path
            The path where the correlation plots (if specified) and
            roi shapefile(s) are saved.
            * if odir=None and plot=True, then raise ValueError
            * if odir=None and roi_shpfile=None then shapefiles
              will be stored in a temporary directory that are
              removed upon completion.

        Returns
        -------
        deglinted_dxr : xarray.Dataset
            xarray object containing the deglinted vis_bands

        Notes
        -----
        1) Python's rasterio and fiona are used to load
           the shapefile as a mask using the NIR/SWIR band
           as the reference.

        2) Individual correlation plots are generated
           between each VIS band and the NIR/SWIR band.

        Raises
        ------
        ValueError
            * if any of the bands do not exist;
            * if odir=None and plot=True;
            * if dwnscale_factor < 1
            * if len(roi_shplist) != number timestamps in xarray;
            * if roi_shplist contains elements that aren't pathlib.Path
              or str objects;
            * if roi_shplist contains shapefiles that do not exist
        """
        if dwnscale_factor < 1:
            raise ValueError("\ndwnscale_factor must be a float >= 1")

        # Check potential input errors
        if not odir and plot:
            raise ValueError("odir=None and plot=True. Please specify odir")

        # create TemporaryDirectory if odir=None. At this point,
        # tmpdir will be used to store the roi shapefiles (if required)
        tmpdir = None
        if not odir:
            tmpdir = tempfile.TemporaryDirectory(".tmp", "hedley-")
            odir = Path(tmpdir.name)

        # Check if variable names in vis_bands & corr_band exist
        self.check_bands_in_xarr(vis_bands)
        self.check_bands_in_xarr([corr_band])

        if not roi_shplist:
            # let the user select the homogeneous roi for each timestamp
            roi_shplist = []
            for t in range(self.ntime):
                # generate quicklook RGB
                rgb_im, rgb_meta = rf.quicklook_rgb_xr(
                    xarr=self.xarr.isel(time=t),
                    rgb_varlist=rgb_varlist,
                    scale_factor=self.scale_factor,
                    dwnscale_factor=dwnscale_factor,
                    mask_nodata=True,
                )

                # select a homogeneous water ROI and store in shapefile
                shp_file = odir / f"user_roi_{t}.shp"
                rf.create_roi_shp(shp_file, rgb_im, rgb_meta)

                roi_shplist.append(shp_file)

        else:
            # check roi_shplist
            self.check_roi_shplist(roi_shplist)

        # initiate plotting variables
        if plot:
            # fig & ax will be reused to reduce memory consumption
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # get the nir/swir band
        corr_im = self.xarr.variables[corr_band].values  # 3D

        # create metadata dict for creating a mask from a shapefile
        metad = {
            "crs": self.xarr.crs,
            "width": self.ncols,
            "height": self.nrows,
            "transform": self.xarr.affine,
        }

        xr_dvars = {}
        for visname in vis_bands:
            vis_im = self.xarr.variables[visname].values  # 3D
            nodata = self.xarr.variables[visname].attrs["nodata"]
            dtype = self.xarr.variables[visname].dtype

            # create empty output array
            deglinted_im = np.zeros(
                [self.ntime, self.nrows, self.ncols], order="C", dtype=dtype
            )

            success_bands = []

            # iterate through time
            for t in range(self.ntime):

                # check for nodata/empty arrays:
                if rf.check_singleval(vis_im[t, :, :], nodata) or rf.check_singleval(
                    corr_im[t, :, :], nodata
                ):
                    deglinted_im[t, :, :] = np.full((self.nrows, self.ncols), nodata)
                    success_bands.append(False)
                    continue  # go to next timestamp

                # water_mask should identify all valid water pixels
                water_mask = (vis_im[t, :, :] != nodata) & (corr_im[t, :, :] != nodata)

                # create mask from the shapefile of this timestamp
                roi_mask = rf.load_mask_from_shp(roi_shplist[t], metad)
                roi_mask[~water_mask] = False

                if plot:
                    vis_bname = f"{visname}_time{t}"
                    corr_bname = f"{corr_band}_time{t}"
                    plot_tuple = (fig, ax, vis_bname, corr_bname, odir)
                else:
                    plot_tuple = None

                deglinted_im[t, :, :], success = hedley_backend(
                    vis_im=vis_im[t, :, :],
                    corr_im=corr_im[t, :, :],
                    water_mask=water_mask,
                    roi_mask=roi_mask,
                    nodata=nodata,
                    scale_factor=self.scale_factor,
                    clip=False,
                    plot=plot,
                    plot_tuple=plot_tuple,
                )

                success_bands.append(success)

            # success_bands indicate which band/timestamps that the
            # deglinting failed - typically as a result of an
            # insufficient number of valid pixels in the ROI
            xr_dvars[f"{visname}"] = xr.Variable(
                dims=["time", "y", "x"],
                data=deglinted_im,
                attrs={"deglint_success": success_bands},
            )

        # create the output xarray.Dataset
        new_attrs = self.xarr.attrs.copy()
        new_attrs["deglint_algorithm"] = "Hedley et al. (2005)"

        new_coords = {
            "time": self.xarr.coords["time"].values,
            "y": self.xarr.coords["y"].values,
            "x": self.xarr.coords["x"].values,
        }

        deglinted_dxr = xr.Dataset(data_vars=xr_dvars, coords=new_coords, attrs=new_attrs)

        # cleanup tmpdir
        if tmpdir is not None:
            tmpdir.cleanup()

        return deglinted_dxr

    def cox_munk(
        self,
        vis_bands: List[str],
        vzen_band: str,
        szen_band: str,
        razi_band: str,
        wind_speed: float = 5,
    ) -> xr.Dataset:
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
        vis_bands : list
            A list of band numbers in the visible that will be deglinted

        vzen_band : str
            The variable name in the xarray.Datatset for the satellite
            view-zenith band

        szen_band : str
            The variable name in the xarray.Dataset for the solar-zenith
            band

        razi_band : str
            The variable name in the xarray.Dataset for the relative
            azimuth band

        wind_speed : float
            wind speed (m/s)

        Returns
        -------
        deglinted_dxr : xr.Dataset
            xarray.Dataset object containing the deglinted vis_bands

        Raises
        ------
        ValueError:
            * If any of the input variable names do not exist
            * if input arrays are not two-dimensional
            * if dimension mismatch
            * if wind_speed < 0
        """
        # check if all input variable names exist in the xarray.Dataset
        self.check_bands_in_xarr(vis_bands)
        self.check_bands_in_xarr([vzen_band])
        self.check_bands_in_xarr([szen_band])
        self.check_bands_in_xarr([razi_band])

        view_zenith = self.xarr.variables[vzen_band].values  # 3D
        solar_zenith = self.xarr.variables[szen_band].values  # 3D
        relative_azimuth = self.xarr.variables[razi_band].values  # 3D

        # for these arrays, nodata = np.nan
        # self.xarr.variables[vzen_band].attrs["nodata"] = "NaN"
        # i.e. a str, convert to np.float64
        p_nodata = np.float64(self.xarr.variables[vzen_band].attrs["nodata"])
        szen_nodata = np.float64(self.xarr.variables[szen_band].attrs["nodata"])
        razi_nodata = np.float64(self.xarr.variables[razi_band].attrs["nodata"])

        xr_dvars = {}
        for visname in vis_bands:
            vis_im = self.xarr.variables[visname].values
            nodata = self.xarr.variables[visname].attrs["nodata"]
            dtype = self.xarr.variables[visname].dtype  # <class 'numpy.dtype'>

            deglinted_im = np.zeros(
                [self.ntime, self.nrows, self.ncols], order="C", dtype=dtype
            )

            success_bands = []
            # iterate through time and deglint
            for t in range(self.ntime):

                # check for nodata/empty arrays:
                if (
                    rf.check_singleval(view_zenith[t, :, :], p_nodata)
                    or rf.check_singleval(solar_zenith[t, :, :], szen_nodata)
                    or rf.check_singleval(relative_azimuth[t, :, :], razi_nodata)
                    or rf.check_singleval(vis_im[t, :, :], nodata)
                ):
                    deglinted_im[t, :, :] = np.full((self.nrows, self.ncols), nodata)
                    success_bands.append(False)
                    continue  # go to next timestamp

                # ------------------------------- #
                #  Estimate sunglint reflectance  #
                # ------------------------------- #

                p_glint, p_fresnel = coxmunk_backend(
                    view_zenith=view_zenith[t, :, :],
                    solar_zenith=solar_zenith[t, :, :],
                    relative_azimuth=relative_azimuth[t, :, :],
                    wind_speed=wind_speed,
                    return_fresnel=False,
                )
                # although the fresnel reflectance (p_fresnel) is not used,
                # it is useful to keep for testing/debugging

                # Notes:
                #    * if return_fresnel=True  then p_fresnel = numpy.ndarray
                #    * if return_fresnel=False then p_fresnel = None
                #    * 0 <= p_glint   (np.float32) <= 1
                #    * 0 <= p_fresnel (np.float32) <= 1

                # In this implementation, the scale_factor (usually 10,000)
                # will not be applied to minimise storage of deglinted
                # bands. Hence, we need to multiply p_glint by this factor
                p_glint[p_glint != p_nodata] *= self.scale_factor

                water_mask = (vis_im[t, :, :] != nodata) & (p_glint != p_nodata)

                deglinted_im[t, :, :] = subtract_backend(
                    vis_band=vis_im[t, :, :],
                    corr_band=p_glint,
                    water_mask=water_mask,
                    nodata=nodata,
                    scale_factor=self.scale_factor,
                    clip=False,
                )
                success_bands.append(True)

            xr_dvars[f"{visname}"] = xr.Variable(
                dims=["time", "y", "x"],
                data=deglinted_im,
                attrs={"deglint_success": success_bands},
            )

        # create the output xarray.Dataset
        new_attrs = self.xarr.attrs.copy()
        new_attrs["deglint_algorithm"] = "wind-direction-independent Cox and Munk (1954)"

        new_coords = {
            "time": self.xarr.coords["time"].values,
            "y": self.xarr.coords["y"].values,
            "x": self.xarr.coords["x"].values,
        }

        deglinted_dxr = xr.Dataset(data_vars=xr_dvars, coords=new_coords, attrs=new_attrs)

        return deglinted_dxr


class GlintCorr:
    """
    yaml and dc_dataset version of Glint Correction
    """

    def __init__(
        self,
        dc_dataset: Union[Dataset, Path, str],
        sub_product: str,
        scale_factor: Union[int, float] = 10000.0,
        fmask_file: Optional[Path] = None,
    ):
        """
        Initiate the sunglint correction class

        Parameters
        ----------
        dc_dataset: datacube.model.Dataset, Path or str
            Datacube dataset object or *_final.odc-metadata.yaml Path

        sub_product: str
            Sub-product name, e.g. "lmbskyg", "nbar", "nbart", "lambertian"

        scale_factor: int or float
            factor used to convert int16 to float32 (default = 10000)

        fmask_file: Path or None
            Path to fmask file

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
            # check if Path is a yaml.
            metadata_dict = self.load_yaml(dc_dataset)
            self.group_path = dc_dataset.parent

        elif isinstance(dc_dataset, str):
            metadata_dict = self.load_yaml(Path(dc_dataset))
            self.group_path = Path(dc_dataset).parent

        else:
            raise ValueError(
                "dc_dataset must be a datacube.model.Dataset, pathlib.Path object or str"
            )

        self.sub_product = sub_product
        self.meas_dict = self.get_meas_dict(metadata_dict)
        self.sensor = self.get_sensor(metadata_dict)

        # self.overpass_datetime isn't used at the moment.
        self.overpass_datetime = self.get_overpass_datetime(metadata_dict)

        # check paths
        self.check_path_exists(self.group_path)

        # check the sensor
        self.check_sensor()

        # extract the res_dict
        self.res_dict = self.create_band_res_dict(metadata_dict)

        # define the scale factor to convert to reflectance
        self.scale_factor = scale_factor

        # get list of tif files from self.meas_dict
        self.band_list, self.band_ids, self.bandnames, self.bandres = self.get_band_list()

        # get a useful output basename
        self.useful_obase = self.output_basename(self.band_list[0])

        # get a useful deep-water ROI shapefil
        self.obase_shp = self.useful_obase + "_deepWater_ROI.shp"

        # get the list of rgb bands that will be used in generating
        # an RGB image for selecting a homogeneous water ROI
        self.rgb_bandlist = self.get_rgb_blist()

        if fmask_file:
            self.fmask_file = Path(fmask_file)
        else:
            # get fmask from self.meas_dict
            self.fmask_file = self.get_fmask_file()

        self.check_path_exists(self.fmask_file)

    def load_yaml(self, p: Path) -> dict:
        if ".odc-metadata.yaml" not in p.name:
            raise ValueError(f"{p} is not a .odc-metadata.yaml")

        with p.open() as f:
            contents = yaml.load(f, Loader=yaml.FullLoader)
        return contents

    def output_basename(self, filename: Path) -> str:
        """ Get a useful output basename """
        return filename.stem.split("_final")[0]

    def deglint_ofile(
        self, spatial_res: Union[float, int], out_dir: Path, visible_bfile: Path
    ) -> Path:
        """ Get deglint filename """
        vis_bname = visible_bfile.stem
        return out_dir / "{0}-deglint-{1}m.tif".format(vis_bname, spatial_res)

    def check_sensor(self):
        if self.sensor not in SUPPORTED_SENSORS:
            msg = f"Supported sensors are: {SUPPORTED_SENSORS}, recieved {self.sensor}"
            raise ValueError(msg)

    def check_path_exists(self, path: Path):
        """ Checks if a path exists """
        if not path.exists():
            raise Exception("\nError: {0} does not exist.".format(path))

    def check_bandnum_exist(self, bandnums: List[str], required_bandnums: List[str]):
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

    def create_band_res_dict(self, metadata_dict: dict) -> dict:
        """
        Create a dictionary that contains the spatial
        resolution for each band present in self.meas_dict.

        Parameters
        ----------
        metadata_dict : dict
            metadata dictionary

        Returns
        -------
        band_res : dict
            e.g. if S2A/B
            {
                'nbar_blue': 10.0,
                'nbar_coastal_aerosol': 60.0,
                'nbar_contiguity': 20.0,
            }
        """
        if "grids" in metadata_dict:
            # new style metadata document
            grid_dict = metadata_dict["grids"]
            res_dict = dict()
            for bname in self.meas_dict:
                if "grid" in self.meas_dict[bname]:
                    gridname = self.meas_dict[bname]["grid"]
                else:
                    gridname = "default"

                res_dict[bname] = grid_dict[gridname]["transform"][0]

        else:
            # old style metadata document
            res_dict = dict()
            for bname in self.meas_dict:
                res_dict[bname] = self.meas_dict[bname]["info"]["geotransform"][1]

        return res_dict

    def create_res_ordered_metad(self, vis_bands: List[str]) -> dict:
        """
        Create a dictionary that groups the input band
        paths based on their spatial resolutions

        Parameters
        ----------
        vis_bands : list
            The list of visible bands that will be deglinted

        Returns
        -------
        res_ordered: dict
            A resolution ordered dictionary, e.g.

            if S2A/B, vis_bands = ["1", "2", "3", "4", "5", "6"]
            res_ordered = {
                10.0: {
                    "blue": ("2", Path(/path/to/blue.tif)),
                    "green": ("3", Path(/path/to/green.tif)),
                    "red": ("4", Path(/path/to/red.tif)),
                }
                20.0: {
                    "red_edge_1": ("5", Path(/path/to/red_edge_1.tif)),
                    "red_edge_2": ("6", Path(/path/to/red_edge_2.tif)),
                }
                60.0: {
                    "coastal_aerosol": ("1", Path(/path/to/coastal_aerosol.tif)),
                }
            }

            if S2A/B, vis_bands = ["1"]
            res_ordered = {
                60.0: {
                    "coastal_aerosol": ("1", Path(/path/to/coastal_aerosol.tif)
                }
            }

        """
        # initialise a nested dictionary
        res_ordered = dict()
        # Get the resolution of the vis_bands
        for visband in vis_bands:
            ix_vis = self.band_ids.index(visband)
            res_ordered[self.bandres[ix_vis]] = dict()

        for z in range(0, len(vis_bands)):
            ix_vis = self.band_ids.index(vis_bands[z])
            res_ordered[self.bandres[ix_vis]][self.bandnames[ix_vis]] = (
                vis_bands[z],
                self.band_list[ix_vis],
            )

        return res_ordered

    def find_file(self, keyword: str) -> Path:
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

    def get_meas_dict(self, metadata_dict: dict) -> dict:
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
                raise ValueError(
                    'unable to extract bands, "bands" not in metadata_dict["image"]'
                )

        elif "measurements" in metadata_dict:
            # newer version
            meas_dict = metadata_dict["measurements"]
        else:
            raise ValueError(
                "neither image nor measurements keys were found in metadata_dict"
            )

        return meas_dict

    def get_sensor(self, metadata_dict: dict) -> str:
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

        def join_sensor(sensor_name):
            # if sensor_name = "sentinel-2a" or "sentinel_2a"
            # then the return is sentinel2a
            if "_" in sensor_name:
                return "".join(sensor_name.split("_"))

            elif "-" in sensor_name:
                return "".join(sensor_name.split("-"))

            else:
                return sensor_name

        msg = "Unable to extract sensor name"
        if "platform" in metadata_dict:
            # older version
            try:
                sensor = join_sensor(metadata_dict["platform"]["code"].lower())
            except KeyError:
                raise ValueError(f'{msg}, "code" not in metadata_dict["platform"]')

        elif "properties" in metadata_dict:
            # newer version
            try:
                sensor = join_sensor(metadata_dict["properties"]["eo:platform"].lower())
            except KeyError:
                raise ValueError(
                    f'{msg}, "eo:platform" not in metadata_dict["properties"]'
                )

        else:
            raise ValueError(
                "neither platform nor properties keys were found in metadata_dict"
            )

        return sensor

    def get_overpass_datetime(self, metadata_dict: dict) -> datetime:
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

            raise ValueError(f"{msg}")

        return overpass_datetime

    def get_band_list(self) -> Tuple[List[str], List[str], List[str], List[str]]:
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

        bandres : list of floats/ints
            A list of the spatial resolutions

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
        bandres = []
        for key in self.meas_dict.keys():
            key_spl = key.split("_")
            if key_spl[0].find(self.sub_product) != -1:
                basename = self.meas_dict[key]["path"]

                # skip over files that aren't needed for deglinting
                if basename.lower() in skip_bands:
                    continue

                # using the band name and res_dict, get the spatial resolution
                bandres.append(self.res_dict[key])

                # Get the resolution from the grid_dict
                bfile = self.group_path.joinpath(basename)

                # ensure that bfile exists, raise exception if not
                self.check_path_exists(bfile)

                bnum = os.path.splitext(basename)[0][-2:].strip("0")

                bandnames.append("_".join(key_spl[1:]))
                bandnums.append(bnum)
                bandlist.append(bfile)

        if not bandlist:
            raise Exception("Could not find any geotifs in '{0}'".format(self.group_path))

        return bandlist, bandnums, bandnames, bandres

    def get_rgb_blist(self) -> List[Path]:
        """
        Get the RGB band-list
        """
        if (
            (self.sensor == "sentinel2a")
            or (self.sensor == "sentinel2b")
            or (self.sensor == "landsat8")
        ):
            ix_red = self.band_ids.index("4")
            ix_grn = self.band_ids.index("3")
            ix_blu = self.band_ids.index("2")

        if (self.sensor == "landsat5") or (self.sensor == "landsat7"):
            ix_red = self.band_ids.index("3")
            ix_grn = self.band_ids.index("2")
            ix_blu = self.band_ids.index("1")

        if self.sensor == "worldview2":
            ix_red = self.band_ids.index("5")
            ix_grn = self.band_ids.index("3")
            ix_blu = self.band_ids.index("2")

        return [self.band_list[ix_red], self.band_list[ix_grn], self.band_list[ix_blu]]

    def get_fmask_file(self) -> Path:
        """
        Get the fmask file from self.meas_dict

        Returns
        -------
        fmask_file : Path
            fmask filename

        Raises
        ------
        ValueError if fmask file is not found
        """
        msg = "Unable to extract fmask"
        if "fmask" in self.meas_dict:
            if "path" in self.meas_dict["fmask"]:
                fmask_file = self.group_path.joinpath(self.meas_dict["fmask"]["path"])
            else:
                raise ValueError(f'{msg}, "path" not in self.meas_dict["fmask"]')

        elif "oa_fmask" in self.meas_dict:
            if "path" in self.meas_dict["oa_fmask"]:
                fmask_file = self.group_path.joinpath(self.meas_dict["oa_fmask"]["path"])
            else:
                raise ValueError(f'{msg}, "path" not in self.meas_dict["oa_fmask"]')

        else:
            raise ValueError(f'{msg}, "fmask" or "oa_fmask" not in self.meas_dict')

        return fmask_file

    def glint_subtraction(
        self,
        vis_bands: List[str],
        corr_band: str,
        water_val: int = 5,
    ) -> List[xr.Dataset]:
        """
        This sunglint correction assumes that glint reflectance
        is nearly spectrally flat in the VIS-NIR. Hence, the NIR/SWIR
        reflectance is subtracted from the VIS bands.

        Dierssen, H.M., Chlus, A., Russell, B. 2015. Hyperspectral
        discrimination of floating mats of seagrass wrack and the
        macroalgae Sargassum in coastal waters of Greater Florida
        Bay using airborne remote sensing. Remote Sens. Environ.,
        167(15), 247-258, doi: https://doi.org/10.1016/j.rse.2015.01.027


        Parameters
        ----------
        vis_bands : list
            A list of band numbers in the visible that will be deglinted

        corr_band : str
            The NIR/SWIR band number used to deglint the VIS bands in vis_bands

        water_val : int
            The fmask value for water pixels (default = 5)

        Returns
        -------
        dxr_list : list
            A list of xarrays, where each xarray contains
            deglinted bands of the same resolution

        Notes
        -----
        1) fmask file is resampled to spatial resolution
           to the VIS bands using Rasterio
           (Resampling.bilinear)

        Raises
        ------
        ValueError
            * if any of the bands do not exist
            * if any of the input data only contains nodata
        """

        # Check that the input vis bands exist
        self.check_bandnum_exist(self.band_ids, vis_bands)
        self.check_bandnum_exist(self.band_ids, [corr_band])

        ix_corr = self.band_ids.index(corr_band)
        corr_bandpath = self.band_list[ix_corr]  # Path

        # ------------------------------ #
        # Group the input bands based on #
        #    their spatial resolution    #
        # ------------------------------ #
        res_ordered_vis = self.create_res_ordered_metad(vis_bands)  # dict

        # ------------------------------ #
        #  Iterate over all spatial res. #
        #  thus creating  an xarray for  #
        #        each spatial res.       #
        # ------------------------------ #
        dxr_list = []
        for res in res_ordered_vis:
            # initialise the xarray data dict.
            xr_dvars = {}

            # iterate over all input bands at this given res.
            for i, bname in enumerate(res_ordered_vis[res]):
                vis_bandpath = res_ordered_vis[res][bname][1]

                # ------------------------------ #
                #        load visible band       #
                # ------------------------------ #
                with rasterio.open(vis_bandpath, "r") as ds_vis:
                    vis_im = ds_vis.read(1)
                    vis_meta = ds_vis.meta.copy()
                    nodata = ds_vis.nodata

                    if nodata is not None:
                        if rf.check_singleval(vis_im, nodata):
                            raise ValueError(
                                f"vis_im only contains a single value ({nodata})"
                            )

                    # Because we are iterating over visible bands that
                    # have the same spatial resolution, crs and Affine
                    # transformation, we only need resample the fmask
                    # and NIR once.
                    if i == 0:
                        # Resample and load fmask_file
                        fmask = rf.resample_file_to_ds(
                            self.fmask_file, ds_vis, Resampling.mode
                        )

                        # Resample NIR/SWIR band
                        corr_im = rf.resample_file_to_ds(
                            corr_bandpath, ds_vis, Resampling.bilinear
                        )

                        if rf.check_singleval(corr_im, nodata):
                            raise ValueError(
                                f"corr_im only contains a single value ({nodata})"
                            )

                # create mask and deglint
                water_mask = (
                    (fmask == water_val) & (vis_im != nodata) & (corr_im != nodata)
                )

                deglint_band = subtract_backend(
                    vis_band=vis_im,
                    corr_band=corr_im,
                    water_mask=water_mask,
                    nodata=nodata,
                    scale_factor=self.scale_factor,
                    clip=False,
                )

                # add 3D array (1, nrows, ncols) to xarray dict.
                data_varname = f"{self.sub_product}_{bname}"
                xr_dvars[data_varname] = xr.Variable(
                    dims=["time", "y", "x"],
                    data=deglint_band.reshape(1, vis_meta["height"], vis_meta["width"]),
                )

            # end-for i, bname
            dxr = create_xr(
                xr_dvars,
                vis_meta,
                self.overpass_datetime,
                "NIR/SWIR subtraction",
            )

            dxr_list.append(dxr)

        # endfor res
        return dxr_list

    def hedley_2005(
        self,
        vis_bands: List[str],
        corr_band: str,
        water_val: int = 5,
        plot: bool = False,
        roi_shpfile: Optional[Path] = None,
        dwnscale_factor: Optional[float] = 3.0,
        odir: Optional[Path] = None,
    ) -> List[xr.Dataset]:
        """
        Sunglint correction using the algorithm:
        Hedley, J. D., Harborne, A. R., Mumby, P. J. (2005). Simple and
        robust removal of sun glint for mapping shallow-water benthos.
        International Journal of Remote Sensing, 26(10), 2107-2112.

        Parameters
        ----------
        vis_bands : list
            A list of band numbers in the visible that will be deglinted

        corr_band : str
            The NIR/SWIR band number used to deglint the VIS bands in vis_bands

        water_val : int
            The fmask value for water pixels (default = 5)

        plot : bool (True | False)
            True will save the correlation plots to the odir specified above.

        roi_shpfile : str
            Path to shapefile containing a polygon of a deep
            water region containing a range of sunglint
            contaminated pixels

        dwnscale_factor : float >= 1
            The downscaling factor used to downscale the native RGB
            bands to generate the quicklook RGB.

            If dwnscale_factor = 3 then the spatial resolution of the
            quicklook RGB will be reduced by a factor of three

            If dwnscale_factor = 1 then no downscaling is performed

        odir : str
            The path where the correlation plots (if specified) are saved.
            if odir=None  tempfile is used

        Returns
        -------
        dxr_list : list
            A list of xarrays, where each xarray contains
            deglinted bands of the same resolution

        Notes
        -----
        1) fmask file is resampled to spatial resolution
           to the VIS bands using Rasterio
           (Resampling.bilinear)

        2) Python's rasterio and fiona are used to load
           the shapefile as a mask using the NIR/SWIR band
           as the reference.

        3) Individual correlation plots are generated
           between each VIS band and the NIR/SWIR band.

        4) NIR/SWIR band is upscaled/downscaled to the
           vis bands.

        Raises
        ------
        ValueError
            * if any of the bands do not exist
            * if odir=None and plot=True
            * if dwnscale_factor < 1
        """
        # Check potential input errors
        if not odir and plot:
            raise ValueError("odir=None and plot=True. Please specify odir")

        if dwnscale_factor < 1:
            raise ValueError("\ndwnscale_factor must be a float >= 1")

        # create TemporaryDirectory if odir=None. At this point,
        # tmpdir will be used to store the roi shapefiles (if required)
        tmpdir = None
        if not odir:
            tmpdir = tempfile.TemporaryDirectory(".tmp", "hedley-")
            odir = Path(tmpdir.name)
        else:
            if isinstance(odir, str):
                odir = Path(odir)

            if not odir.exists():
                odir.mkdir(exist_ok=True)

        # check roi shapefile
        if not roi_shpfile:
            # if not supplied, then create a shapefile in odir
            roi_shpfile = odir / "deepWater_ROI_polygon.shp"

        else:
            if isinstance(roi_shpfile, str):
                roi_shpfile = Path(roi_shpfile)

        if not roi_shpfile.exists():
            # Generate downsampled quicklook RGB
            rgb_im, rgb_meta = rf.quicklook_rgb(
                rgb_bandlist=self.rgb_bandlist,
                scale_factor=self.scale_factor,
                dwnscale_factor=dwnscale_factor,
            )

            # use interactive mode to generate the shapefile
            rf.create_roi_shp(roi_shpfile, rgb_im, rgb_meta)

        # initiate plot if specified
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # Check that the input vis bands exist
        self.check_bandnum_exist(self.band_ids, vis_bands)
        self.check_bandnum_exist(self.band_ids, [corr_band])

        ix_corr = self.band_ids.index(corr_band)
        corr_bandpath = self.band_list[ix_corr]  # Path

        # ------------------------------ #
        # Group the input bands based on #
        #    their spatial resolution    #
        # ------------------------------ #
        res_ordered_vis = self.create_res_ordered_metad(vis_bands)  # dict

        # ------------------------------ #
        #  Iterate over all spatial res. #
        #  thus creating  an xarray for  #
        #        each spatial res.       #
        # ------------------------------ #
        dxr_list = []
        for res in res_ordered_vis:
            # initialise the xarray data dict.
            xr_dvars = {}

            # iterate over all input bands at this given res.
            for i, bname in enumerate(res_ordered_vis[res]):
                vis_bandpath = res_ordered_vis[res][bname][1]

                # ------------------------------ #
                #        load visible band       #
                # ------------------------------ #
                with rasterio.open(vis_bandpath, "r") as ds_vis:
                    vis_im = ds_vis.read(1)
                    vis_meta = ds_vis.meta.copy()
                    nodata = ds_vis.nodata

                    if nodata is not None:
                        if rf.check_singleval(vis_im, nodata):
                            raise ValueError(
                                f"vis_im only contains a single value ({nodata})"
                            )

                    # Because we are iterating over visible bands that
                    # have the same spatial resolution, crs and Affine
                    # transformation, we only need resample the fmask
                    # and corr_im once.
                    if i == 0:
                        # Resample and load fmask_file
                        fmask = rf.resample_file_to_ds(
                            self.fmask_file, ds_vis, Resampling.mode
                        )

                        # Resample NIR/SWIR band
                        corr_im = rf.resample_file_to_ds(
                            corr_bandpath, ds_vis, Resampling.bilinear
                        )

                        if rf.check_singleval(corr_im, nodata):
                            raise ValueError(
                                f"corr_im only contains a single value ({nodata})"
                            )

                        # Load shapefile as a mask
                        roi_mask = rf.load_mask_from_shp(roi_shpfile, vis_meta)

                # create masks
                flag_mask = (
                    (fmask != water_val) | (vis_im == nodata) | (corr_im == nodata)
                )
                water_mask = ~flag_mask
                roi_mask[flag_mask] = False

                # ------------------------------ #
                #       Sunglint Correction      #
                # ------------------------------ #
                if plot:
                    vis_bname = f"B{res_ordered_vis[res][bname][0]}"
                    plot_tuple = (fig, ax, vis_bname, f"B{corr_band}", odir)
                else:
                    plot_tuple = None

                deglint_band, success = hedley_backend(
                    vis_im=vis_im,
                    corr_im=corr_im,
                    water_mask=water_mask,
                    roi_mask=roi_mask,
                    nodata=nodata,
                    scale_factor=self.scale_factor,
                    clip=False,
                    plot=plot,
                    plot_tuple=plot_tuple,
                )

                # add 3D array (1, nrows, ncols) to xarray dict.
                data_varname = f"{self.sub_product}_{bname}"
                xr_dvars[data_varname] = xr.Variable(
                    dims=["time", "y", "x"],
                    data=deglint_band.reshape(1, vis_meta["height"], vis_meta["width"]),
                )

            # end-for i, bname
            dxr = create_xr(
                xr_dvars,
                vis_meta,
                self.overpass_datetime,
                "Hedley et al. (2005)",
            )

            dxr_list.append(dxr)

        # cleanup
        if tmpdir is not None:
            tmpdir.cleanup()

        # endfor res
        return dxr_list

    def cox_munk(
        self,
        vis_bands: List[str],
        vzen_file: Union[Path, None] = None,
        szen_file: Union[Path, None] = None,
        razi_file: Union[Path, None] = None,
        wind_speed: float = 5,
        water_val: int = 5,
    ) -> List[xr.Dataset]:
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
        vis_bands : list
            A list of band numbers in the visible that will be deglinted

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
        dxr_list : list
            A list of xarrays, where each xarray contains
            deglinted bands of the same resolution

        Raises
        ------
        ValueError:
            * if input arrays are not two-dimensional
            * if any input arrays only contain nodata
            * if dimension mismatch
            * if wind_speed < 0
        """
        # --- check vzen_file --- #
        if vzen_file:
            if not isinstance(vzen_file, Path):
                vzen_file = Path(vzen_file)

            self.check_path_exists(vzen_file)

        else:
            # find view zenith from self.meas_dict
            vzen_file = self.find_file("satellite_view")

        # --- check szen_file --- #
        if szen_file:
            if not isinstance(szen_file, Path):
                szen_file = Path(szen_file)

            self.check_path_exists(szen_file)

        else:
            # find solar zenith from self.meas_dict
            szen_file = self.find_file("solar_zenith")

        # --- check razi_file --- #
        if razi_file:
            if not isinstance(razi_file, Path):
                razi_file = Path(razi_file)

            self.check_path_exists(razi_file)

        else:
            # find relative azimuth from self.meas_dict
            razi_file = self.find_file("relative_azimuth")

        # Check that the input vis bands exist
        self.check_bandnum_exist(self.band_ids, vis_bands)

        # ------------------------------- #
        #  Estimate sunglint reflectance  #
        # ------------------------------- #
        vzen_im, vzen_meta = rf.load_singleband(vzen_file)
        szen_im, szen_meta = rf.load_singleband(szen_file)
        razi_im, razi_meta = rf.load_singleband(razi_file)
        cm_meta = szen_meta.copy()

        # for these arrays, nodata = np.nan
        if rf.check_singleval(vzen_im, vzen_meta["nodata"]):
            raise ValueError("vzen_im only contains a single value")

        if rf.check_singleval(szen_im, szen_meta["nodata"]):
            raise ValueError("szen_im only contains a single value")

        if rf.check_singleval(razi_im, razi_meta["nodata"]):
            raise ValueError("razi_im only contains a single value")

        # cox and munk:
        p_glint, p_fresnel = coxmunk_backend(
            view_zenith=vzen_im,
            solar_zenith=szen_im,
            relative_azimuth=razi_im,
            wind_speed=wind_speed,
            return_fresnel=False,
        )
        # although the fresnel reflectance (p_fresnel) is not used,
        # it is useful to keep for testing/debugging

        p_nodata = cm_meta["nodata"]  # this is np.nan

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
        # Group the input bands based on #
        #    their spatial resolution    #
        # ------------------------------ #
        res_ordered_vis = self.create_res_ordered_metad(vis_bands)  # dict

        # ------------------------------ #
        #  Iterate over all spatial res. #
        #  thus creating  an xarray for  #
        #        each spatial res.       #
        # ------------------------------ #
        dxr_list = []
        for res in res_ordered_vis:
            # initialise the xarray data dict.
            xr_dvars = {}

            # iterate over all input bands at this given res.
            for i, bname in enumerate(res_ordered_vis[res]):
                vis_bandpath = res_ordered_vis[res][bname][1]

                # ------------------------------ #
                #        load visible band       #
                # ------------------------------ #
                with rasterio.open(vis_bandpath, "r") as ds_vis:
                    vis_im = ds_vis.read(1)
                    vis_meta = ds_vis.meta.copy()
                    nodata = ds_vis.nodata

                    if nodata is not None:
                        if rf.check_singleval(vis_im, nodata):
                            raise ValueError(
                                f"vis_im only contains a single value ({nodata})"
                            )

                    # Because we are iterating over visible bands that
                    # have the same spatial resolution, crs and Affine
                    # transformation, we only need resample the fmask
                    # and p_glint once.
                    if i == 0:
                        # Resample and load fmask_file
                        fmask = rf.resample_file_to_ds(
                            self.fmask_file, ds_vis, Resampling.mode
                        )

                        if p_nodata != nodata:
                            p_glint[p_glint == p_nodata] = nodata
                            if isinstance(p_fresnel, np.ndarray):
                                p_fresnel[p_fresnel != p_nodata] *= self.scale_factor
                                p_fresnel[p_fresnel == p_nodata] = nodata
                                # convert from np.float32 to np.int16
                                p_fresnel.astype(dtype=vis_im.dtype)

                        # resample p_glint
                        p_glint_res = rf.resample_band_to_ds(
                            p_glint, cm_meta, ds_vis, Resampling.bilinear
                        )

                # ------------------------------ #
                #       Sunglint Correction      #
                # ------------------------------ #
                water_mask = (fmask == water_val) & (vis_im != nodata)

                deglint_band = subtract_backend(
                    vis_band=vis_im,
                    corr_band=p_glint_res,
                    water_mask=water_mask,
                    nodata=nodata,
                    scale_factor=self.scale_factor,
                    clip=False,
                )

                # add 3D array (1, nrows, ncols) to xarray dict.
                data_varname = f"{self.sub_product}_{bname}"
                xr_dvars[data_varname] = xr.Variable(
                    dims=["time", "y", "x"],
                    data=deglint_band.reshape(1, vis_meta["height"], vis_meta["width"]),
                )

            # end-for i, bname
            dxr = create_xr(
                xr_dvars,
                vis_meta,
                self.overpass_datetime,
                "wind-direction-independent Cox and Munk (1954)",
            )
            dxr_list.append(dxr)

        # endfor res
        return dxr_list
