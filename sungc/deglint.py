#!/usr/bin/env python3

import os
import sys
import pathlib
import rasterio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from os.path import join as pjoin
from rasterio.warp import Resampling

import sungc.rasterio_funcs as rio_funcs
from sungc.interactive import ROI_Selector
from sungc.cox_munk_funcs import cm_sunglint
from sungc.visualise import enhanced_RGB_stretch, display_image

get_bandNums_int = lambda f: int(os.path.splitext(f)[0].split("-")[1])
get_bandNums_str = lambda f: os.path.splitext(f)[0].split("-")[1]


class GlintCorr:
    def __init__(self, dc_dataset, product, fmask_file=None):
        """
        Initiate the sunglint correction class

        Parameters
        ----------
        dc_dataset : datacube.model.Dataset
            Datacube dataset object

        product : str
            Product name 

        Examples
        --------
        """
        self.product = product
        prop_dict = dc_dataset.metadata_doc["properties"]
        meas_dict = dc_dataset.metadata_doc["measurements"]
        base_dir = dc_dataset.local_path.parent

        self.sensor = prop_dict["eo:platform"].lower()
        self.group_path = dc_dataset.local_path.parent
        self.overpass_datetime = prop_dict["datetime"]  # str

        # check paths
        self.check_path_exists(self.group_path)

        # check the sensor
        self.check_sensor()

        # define the scale factor to convert to reflectance
        self.scale_factor = 10000.0

        # get list of tif files
        self.bandList, self.band_ids, self.bandNames = self.get_band_list(meas_dict)

        # get a useful output basename
        # that a user may want to use
        self.useful_obase = self.output_basename(self.bandList[0])

        # get a useful deep-water ROI shapefil
        self.obase_shp = self.useful_obase + "_deepWater_ROI.shp"

        if fmask_file:
            self.fmask_file = pathlib.PosixPath(fmask_file)
        else:
            self.fmask_file = self.group_path.joinpath(meas_dict["oa_fmask"]["path"])

        self.check_path_exists(self.fmask_file)

    def output_basename(self, filename):
        """ Get a useful output basename """
        return filename.stem.split("_final")[0]

    def deglint_ofile(self, spatialRes, out_dir, visible_bfile):
        """ Get deglint filename """
        vis_bname = os.path.basename(os.path.splitext(visible_bfile)[0])
        return pjoin(
            out_dir, "{0}-deglint-{1}m.tif".format(vis_bname, spatialRes),
        )

    def check_sensor(self):
        if (
            (not self.sensor == "sentinel-2a")
            and (not self.sensor == "sentinel-2b")
            and (not self.sensor == "landsat-5")
            and (not self.sensor == "landsat-7")
            and (not self.sensor == "landsat-8")
            and (not self.sensor == "worldview-2")
        ):
            raise Exception(
                'sensor must be "sentinel-2", "landsat-5",'
                '"landsat-7", "landsat-8" or "worldview-2"'
            )

    def check_path_exists(self, path):
        """ Checks if a path exists """
        if not path.exists():
            raise Exception("\nError: {0} does not exist.".format(path))

    def check_bandNum_exist(self, bandNums, required_bandNums):
        """
        Checks if the band numbers in the required list exist

        Parameters
        ----------
        bandNums : list
            A list of band numbers

        required_bandNums : list
            A list of required band numbers

        Raises
        ------
        Exception
            if any required band numbers are missing
            from bandNums
        """
        for req_bn in required_bandNums:
            if not (req_bn in bandNums):
                raise Exception(
                    "B{0} is missing from bands [{1}]".format(
                        req_bn, ", ".join(["B" + str(i) for i in bandNums]),
                    )
                )

    def find_file(self, path, keyword):
        """
        find a file in a directory (path) that
        has a keyword in its filename

        Parameters
        ----------
        path : str
            Directory to search

        keyword : str
            keyword

        Returns
        -------
        filename : None or str
            filename
            if None then file with keyword not found
        """
        filename = None
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.find(keyword) != -1:
                    filename = pjoin(root, f)

        return filename

    def get_band_list(self, meas_dict):
        """
        Get a list of tifs in self.group_path

        Parameters
        ----------
        meas_dict : dict()
            A dictionary containing product band names

        Returns
        -------
        bandList : list
            A list of pathlib

        bandNums : list of strings
            A list of band numbers

        bandNames : list of string
            A list of band names
        """
        bandList = []
        bandNums = []
        bandNames = []
        for key in meas_dict.keys():
            key_spl = key.split("_")
            if key_spl[0].find(self.product) != -1:
                basename = meas_dict[key]["path"]
                bnum = os.path.splitext(basename)[0][-2:].strip("0")

                bandNames.append("_".join(key_spl[1:]))
                bandNums.append(bnum)
                bandList.append(self.group_path.joinpath(basename))

        if not bandList:
            raise Exception(
                "Could not find any geotifs in '{0}'".format(self.group_path)
            )
        return bandList, bandNums, bandNames

    def get_fmask(self):
        """
        Find the fmask.img file in the specified directory
        """
        fmask_file = None
        for root, dirs, files in os.walk(self.group_path):
            for f in files:
                f_indir = pjoin(self.group_path, f)
                if os.path.isfile(f_indir) and f.lower().endswith(
                    ".fmask.img"
                ):
                    fmask_file = f_indir
                    break

        if not fmask_file:
            raise Exception(
                "\nfmask not found in '{0}'".format(self.group_path)
            )

        return fmask_file

    def quicklook_RGB(self, dwnscale_factor=3):
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
            [nRows, nCols, 3] for the three channels

        rgb_meta : dict
            Metadata dictionary taken from rasterio

        Raises
        ------
        Exception
            * if dwnscale_factor < 1
        """
        if dwnscale_factor < 1:
            raise Exception("\ndwnscale_factor must be a float >= 1")

        if (self.sensor == "sentinel-2a") or (self.sensor == "sentinel-2b") or (self.sensor == "landsat-8"):
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

        rgb_bandList = [
            self.bandList[ix_red],
            self.bandList[ix_grn],
            self.bandList[ix_blu],
        ]
        with rasterio.open(rgb_bandList[0], "r") as ds:
            ql_spatialRes = dwnscale_factor * float(ds.transform.a)

        if dwnscale_factor > 1:
            # resample to quicklook spatial resolution
            resmpl_tifs, refl_im, rio_meta = rio_funcs.resample_bands(
                rgb_bandList,
                ql_spatialRes,
                Resampling.nearest,
                load=True,
                save=False,
                odir=None,
            )
        else:
            refl_im, rio_meta = rio_funcs.load_bands(
                rgb_bandList, self.scale_factor, False
            )

        # resample fmask
        resmpl_tifs, fmask_im, fmask_meta = rio_funcs.resample_bands(
            [self.fmask_file],
            ql_spatialRes,
            Resampling.mode,
            load=True,
            save=False,
            odir=None,
        )

        # create a pretty RGB
        rgb_im = enhanced_RGB_stretch(
            refl_im, fmask_im, [0, 1, 2], rio_meta["nodata"], 0.1, 99.9
        )

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

        """

        # generate a quicklook at 20 m spatial resolution
        rgb_im, rgb_meta = self.quicklook_RGB(dwnscaling_factor)

        # let the user select a ROI from the 10m RGB
        ax = display_image(rgb_im, None)
        mc = ROI_Selector(ax=ax)
        mc.interative()
        plt.show()

        # write a shapefile
        mc.verts_to_shp(metadata=rgb_meta, shp_file=shp_file)

        # close the ROI_Selector
        mc = None

    def gao_2007(
        self,
        vis_band_ids,
        nir_band_id,
        odir=None,
        waterVal=5,
    ):
        """
        This sunglint correction assumes that glint reflectance
        is nearly spectrally flat in the VIS-NIR. Hence, the NIR
        reflectance is subtracted from the VIS bands

        Parameters
        ----------
        vis_band_ids : list
            A list of band numbers in the visible that will be deglinted

        nir_band_id : str
            The NIR band number used to deglint the VIS bands in vis_band_ids

        odir : str
            The path where the deglinted geotiff bands are saved.

            if None then:
            odir = pjoin(self.group_path, "DEGLINT", "GAO")

        waterVal : int
            The fmask value for water pixels (default = 5)

        Returns
        -------
        deglint_BandList : list
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
        """
        # create output directory
        if not odir:
            odir = pjoin(self.group_path, "DEGLINT", "GAO")

        if not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)

        # Check that the input vis bands exist
        self.check_bandNum_exist(self.band_ids, vis_band_ids)
        self.check_bandNum_exist(self.band_ids, [nir_band_id])

        ix_nir = self.band_ids.index(nir_band_id)
        nir_bandPath = str(self.bandList[ix_nir])

        # ------------------------------ #
        #       load the NIR band        #
        # ------------------------------ #
        with rasterio.open(nir_bandPath, "r") as nir_ds:
            nir_nRows = nir_ds.height
            nir_nCols = nir_ds.width
            nir_im = nir_ds.read(1)

        # ------------------------------ #
        nBands = len(vis_band_ids)
        deglint_BandList = []

        # Iterate over each visible band
        for z in range(0, nBands):

            ix_vis = self.band_ids.index(vis_band_ids[z])
            vis_bandPath = str(self.bandList[ix_vis])

            # ------------------------------ #
            #        load visible band       #
            # ------------------------------ #
            with rasterio.open(vis_bandPath, "r") as dsVIS:
                vis_im = dsVIS.read(1)

                # get metadata and load NIR array
                kwargs = dsVIS.meta.copy()
                nodata = kwargs["nodata"]
                spatialRes = int(abs(kwargs["transform"].a))

                # ------------------------------ #
                #  Resample and load fmask_file  #
                # ------------------------------ #
                fmask = rio_funcs.resample_band_to_ds(
                    self.fmask_file, dsVIS, Resampling.mode
                )

                # ------------------------------ #
                #       Resample NIR band        #
                # ------------------------------ #
                if (nir_nRows != dsVIS.height) or (nir_nCols != dsVIS.width):
                    # resample the NIR band to match the VIS band
                    nir_im = rio_funcs.resample_band_to_ds(
                        nir_bandPath, dsVIS, Resampling.bilinear
                    )

            # ------------------------------ #
            #       Sunglint Correction      #
            # ------------------------------ #
            # copy band
            deglint_band = np.array(vis_im, order="K", copy=True)

            waterIx = np.where(
                (fmask == waterVal) & (vis_im != nodata) & (nir_im != nodata)
            )

            # 1. deglint water pixels
            deglint_band[waterIx] = vis_im[waterIx] - nir_im[waterIx]
            deglint_band[(vis_im == nodata) | (nir_im == nodata)] = nodata

            # 2. write geotiff
            deglint_otif = self.deglint_ofile(spatialRes, odir, vis_bandPath)
            with rasterio.open(deglint_otif, "w", **kwargs) as dst:
                dst.write(deglint_band, 1)

            if os.path.exists(deglint_otif):
                deglint_BandList.append(deglint_otif)

        # endfor z
        return deglint_BandList

    def hedley_2005(
        self,
        vis_band_ids,
        nir_band_id,
        roi_shpfile=None,
        overwrite_shp=None,
        odir=None,
        waterVal=5,
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
            odir = pjoin(self.group_path, "DEGLINT", "HEDLEY")

        waterVal : int
            The fmask value for water pixels (default = 5)

        plot : bool (True | False)
            True will save the correlation plots to the odir specified above.

        Returns
        -------
        deglint_BandList : list
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
            roi_shpfile = pjoin(self.group_path, "deepWater_ROI_polygon.shp")

        if not os.path.exists(roi_shpfile):
            # use interactive mode to generate the shapefile
            self.create_roi_shp(roi_shpfile)
        else:
            if overwrite_shp:
                # recreate shapefile
                self.create_roi_shp(roi_shpfile)

        # create output directory
        if not odir:
            odir = pjoin(self.group_path, "DEGLINT", "HEDLEY")

        if not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)

        # initiate plot if specified
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # Check that the input vis bands exist
        self.check_bandNum_exist(self.band_ids, vis_band_ids)
        self.check_bandNum_exist(self.band_ids, [nir_band_id])

        ix_nir = self.band_ids.index(nir_band_id)
        nir_bandPath = str(self.bandList[ix_nir])

        # ------------------------------ #
        #       load the NIR band        #
        # ------------------------------ #
        with rasterio.open(nir_bandPath, "r") as nir_ds:
            nir_nRows = nir_ds.height
            nir_nCols = nir_ds.width
            nir_im = nir_ds.read(1)

        # ------------------------------ #
        nBands = len(vis_band_ids)
        deglint_BandList = []

        # Iterate over each visible band
        for z in range(0, nBands):

            ix_vis = self.band_ids.index(vis_band_ids[z])
            vis_bandPath = str(self.bandList[ix_vis])

            # ------------------------------ #
            #        load visible band       #
            # ------------------------------ #
            with rasterio.open(vis_bandPath, "r") as dsVIS:
                vis_im = dsVIS.read(1)

                # get metadata and load NIR array
                kwargs = dsVIS.meta.copy()
                nodata = kwargs["nodata"]
                spatialRes = int(abs(kwargs["transform"].a))

                # ------------------------------ #
                #       Resample NIR band        #
                # ------------------------------ #
                if (nir_nRows != dsVIS.height) or (nir_nCols != dsVIS.width):
                    # resample the NIR band to match the VIS band
                    nir_im = rio_funcs.resample_band_to_ds(
                        nir_bandPath, dsVIS, Resampling.bilinear
                    )

                # ------------------------------ #
                #  Resample and load fmask_file  #
                # ------------------------------ #
                fmask = rio_funcs.resample_band_to_ds(
                    self.fmask_file, dsVIS, Resampling.mode
                )

                # ------------------------------ #
                #    Load shapefile as a mask    #
                # ------------------------------ #
                roi_mask = rio_funcs.load_mask_from_shp(roi_shpfile, dsVIS)

            flag_mask = (
                (fmask != waterVal) | (vis_im == nodata) | (nir_im == nodata)
            )
            water_mask = ~flag_mask
            roi_mask[flag_mask] = nodata

            # ------------------------------ #
            #       Sunglint Correction      #
            # ------------------------------ #
            # copy band
            deglint_band = np.array(vis_im, order="K", copy=True)

            # 1. Find minimum NIR in the roi polygon
            roi_valIx = np.where(roi_mask != nodata)
            valid_NIR = nir_im[roi_valIx]
            minRefl_NIR = valid_NIR.min()

            # 2. Get correlations between current band and NIR
            x_vals = valid_NIR
            y_vals = vis_im[roi_valIx]
            slope, y_inter, r_val, p_val, std_err = stats.linregress(
                x=x_vals, y=y_vals
            )

            # 3. deglint water pixels
            deglint_band[water_mask] = vis_im[water_mask] - slope * (
                nir_im[water_mask] - minRefl_NIR
            )
            deglint_band[(vis_im == nodata) | (nir_im == nodata)] = nodata

            # 4. write geotiff
            deglint_otif = self.deglint_ofile(spatialRes, odir, vis_bandPath)
            with rasterio.open(deglint_otif, "w", **kwargs) as dst:
                dst.write(deglint_band, 1)

            if os.path.exists(deglint_otif):
                deglint_BandList.append(deglint_otif)

            # ------------------------------ #
            #        Plot correlations       #
            # ------------------------------ #
            if plot:
                # clear previous plot
                ax.clear()
                ann_str = (
                    "R-squared = {0:0.2f}\n"
                    "slope = {1:0.3f}\n"
                    "y-inter = {2:0.3f}".format(r_val ** 2, slope, y_inter)
                )
                bnir = "B" + nir_band_id
                bvis = "B" + vis_band_ids[z]
                png_file = pjoin(
                    odir, "Correlation_{0}_vs_{1}.png".format(bnir, bvis),
                )

                # Randomly select a 50,000 points to avoid plotting
                # a huge amount of points
                n_max = 50000
                mpl.rcParams["agg.path.chunksize"] = n_max
                rnd_ix = np.arange(0, len(valid_NIR))
                if len(valid_NIR) > n_max:
                    np.random.shuffle(rnd_ix)

                (ln1,) = ax.plot(
                    x_vals[rnd_ix],
                    y_vals[rnd_ix],
                    color="k",
                    linestyle="None",
                    marker=".",
                )
                x_range = np.array([x_vals.min(), x_vals.max()])
                (ln2,) = ax.plot(
                    x_range,
                    slope * (x_range) + y_inter,
                    color="r",
                    linestyle="-",
                )
                ann1 = ax.annotate(
                    s=ann_str,
                    xy=(0.7, 0.2),
                    xycoords="axes fraction",
                    fontsize=10,
                )
                ax.set_xlabel(bnir)
                ax.set_ylabel(bvis)

                # save figure
                fig.savefig(
                    png_file,
                    format="png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                    dpi=300,
                )

                ln1.remove()
                ln2.remove()
                ann1.remove()
            # end-if plot
        # endfor z

        return deglint_BandList
    
    def cox_munk(
        self,
        vis_band_ids,
        odir=None,
        vzen_file=None,
        szen_file=None,
        razi_file=None,
        wind_speed=5,
        waterVal=5,
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

        odir : str or None
            The path where the deglinted geotiff bands are saved.

            if None then:
            odir = pjoin(self.group_path, "DEGLINT", "COX_MUNK")

        vzen_file : str or None
            sensor view zenith (rasterio-openable) image file

            * if None then the file containing "satellite-view.tif"
              inside group_path is designated as the vzen_file

        szen_file : str or None
            solar zenith (rasterio-openable) image file

            * if None then the file containing "solar-zenith.tif"
              inside group_path is designated as the szen_file

        razi_file : str or None
            Relative azimuth (rasterio-openable) image file
            Relative azimuth = solar_azimuth - sensor_azimuth

            * if None then the file containing "relative-azimuth.tif"
              inside group_path is designated as the razi_file

        wind_speed : float
            Wind speed (m/s)

        waterVal : int
            The fmask value for water pixels (default = 5)

        Returns
        -------
        deglint_BandList : list
            A list of paths to the deglinted geotiff bands

        Raises
        ------
        Exception:
            * if input arrays are not two-dimensional
            * if dimension mismatch
            * if wind_speed < 0
        """
        # create output directory
        if not odir:
            odir = pjoin(self.group_path, "DEGLINT", "COX_MUNK")

        if not os.path.exists(odir):
            os.makedirs(odir, exist_ok=True)

        # --- check vzen_file --- #
        if vzen_file:
            self.check_path_exists(vzen_file)
        else:
            vzen_file = self.find_file(self.group_path, "satellite-view.tif")

        # --- check szen_file --- #
        if szen_file:
            self.check_path_exists(szen_file)
        else:
            szen_file = self.find_file(self.group_path, "solar-zenith.tif")

        # --- check razi_file --- #
        if razi_file:
            self.check_path_exists(razi_file)
        else:
            razi_file = self.find_file(self.group_path, "relative-azimuth.tif")

        # Check that the input vis bands exist
        self.check_bandNum_exist(self.band_ids, vis_band_ids)

        # ------------------------------- #
        #  Estimate sunglint reflectance  #
        # ------------------------------- #

        # load vzen, szen and razi
        vzen_im, vzen_meta = rio_funcs.load_singleBand(vzen_file)
        szen_im, szen_meta = rio_funcs.load_singleBand(szen_file)
        razi_im, razi_meta = rio_funcs.load_singleBand(razi_file)

        # cox and munk:
        p_glint, p_fresnel = cm_sunglint(
            view_zenith=vzen_im,
            solar_zenith=szen_im,
            relative_azimuth=razi_im,
            wind_speed=wind_speed,
            return_fresnel=False,
        )
        # Notes:
        #    * if return_fresnel=True  then p_fresnel = numpy.ndarray
        #    * if return_fresnel=False then p_fresnel = None
        #    * 0 <= p_glint   (np.float32) <= 1
        #    * 0 <= p_fresnel (np.float32) <= 1

        # In this implementation, the scale_factor (usually 10,000)
        # will not be applied to minimise storage of deglinted
        # bands. Hence, we need to multiply p_glint by this factor
        p_glint *= self.scale_factor  # keep as np.float32

        # ------------------------------ #
        nBands = len(vis_band_ids)
        deglint_BandList = []

        for z in range(0, nBands):

            ix_vis = self.band_ids.index(vis_band_ids[z])
            vis_bandPath = self.bandList[ix_vis]

            # ------------------------------ #
            #        load visible band       #
            # ------------------------------ #
            with rasterio.open(vis_bandPath, "r") as dsVIS:
                vis_im = dsVIS.read(1)

                # get metadata
                kwargs = dsVIS.meta.copy()
                nodata = kwargs["nodata"]
                spatialRes = int(abs(kwargs["transform"].a))

                # ------------------------------ #
                #  Resample and load fmask_file  #
                # ------------------------------ #
                fmask = rio_funcs.resample_band_to_ds(
                    self.fmask_file, dsVIS, Resampling.mode
                )

            # ------------------------------ #
            #       Sunglint Correction      #
            # ------------------------------ #
            # copy band
            deglint_band = np.array(vis_im, order="K", copy=True)
            waterMSK = ((fmask == waterVal) & (vis_im != nodata))

            # 1. deglint water pixels
            deglint_band[waterMSK] = vis_im[waterMSK] - p_glint[waterMSK]

            # 2. write geotiff
            deglint_otif = self.deglint_ofile(spatialRes, odir, vis_bandPath)
            with rasterio.open(deglint_otif, "w", **kwargs) as dst:
                dst.write(deglint_band, 1)

            if os.path.exists(deglint_otif):
                deglint_BandList.append(deglint_otif)

        # endfor z
        return deglint_BandList
