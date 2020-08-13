#!/usr/bin/env python3

import re
import os
import sys
import fiona
import rasterio
import rasterio.mask
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join as pjoin

from scipy import stats
from rasterio.warp import reproject, Resampling
from visualise import enhanced_RGB_stretch, display_two_images
from interactive import ROI_Selector

get_bandNums_int = lambda f: int(os.path.splitext(f)[0].split("-")[1])
get_bandNums_str = lambda f: os.path.splitext(f)[0].split("-")[1]


class GlintCorr:
    def __init__(self, group_path, sensor, fmask_file=None):
        """
        Initiate the sunglint correction class

        Parameters
        ----------
        group_path: str
            Path to S2-MSI/L8-OLI/WV2 folder

        sensor : str
            Sensor = {
                "LANDSAT_5",
                "LANDSAT_7",
                "LANDSAT_8",
                "SENTINEL_2",
                "WORLDVIEW_2"
            }

        fmask_file : str or None
            mask image file (that can be opened with rasterio) that
            classifies pixels as water, land, cloud etc.
            if None then a search for the fmask.img file is performed.

        Examples
        --------

        deglint S2 bands using the Hedley et al. (2019) approach
        >>> g = GlintCorr("/path/to/S2_folder", "sentinel-2")
        >>> deglint_BList, tmp_files = g.deglint(algorithm="hedley")
        >>> # deglint_BList = list of paths to deglinted bands
        >>> # tmp_files = list of path to temporary files created
        """
        self.sensor = sensor.lower()
        self.group_path = group_path
        self.fmask_file = fmask_file

        # check the path
        self.check_path_exists(self.group_path)

        # check the sensor
        self.check_sensor()

        # define the scale factor to convert to reflectance
        self.scale_factor = 10000.0

        # get list of tif files
        self.bandList, self.band_ids = self.get_band_list()

        if self.fmask_file:
            self.check_path_exists(self.fmask_file)
        else:
            self.fmask_file = self.get_fmask()

    def get_basename(self, filename):
        """ Get a file's basename without extension """
        return os.path.basename(os.path.splitext(filename)[0])

    def get_resample_bandName(self, filename, spatialRes):
        """ Get the basename of the resampled band """
        if filename.lower().find("fmask") != -1:
            # fmask has a different naming convention
            out_base = "fmask-resmpl-{0}m.tif".format(spatialRes)
        else:
            out_base = "{0}-resmpl-{1}m.tif".format(
                self.get_basename(filename), spatialRes
            )
        return out_base

    def check_sensor(self):
        if (
            (not self.sensor == "sentinel_2")
            and (not self.sensor == "landsat_5")
            and (not self.sensor == "landsat_6")
            and (not self.sensor == "landsat_7")
            and (not self.sensor == "landsat_8")
            and (not self.sensor == "worldview_2")
        ):
            raise Exception(
                'sensor must be "sentinel_2", "landsat_5",'
                '"landsat_7", "landsat_8" or "worldview-2"'
            )

    def check_path_exists(self, path):
        """ Checks if a path exists """
        if not os.path.exists(path):
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

    def display_img(self, img):
        """
        Displays an image using matplotlib

        Parameters
        ----------
        img : numpy.ndarray
            A 2-dimensional numpy array

        Returns
        -------
        ax : <AxesSubplot> class
           The axes of the matplotlib subplot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img, interpolation="None")
        ax.axis("off")
        return ax

    def get_band_list(self):
        """
        Get a list of tifs in self.group_path

        Returns
        -------
        bandList : list of strings
            A list of paths to band geotifs

        bandNums : list of strings
            A list of band numbers
        """
        bandList = []
        bandNums = []
        for root, dirs, files in os.walk(self.group_path):
            for f in files:
                if f.lower().endswith(".tif"):
                    f_split = os.path.splitext(f)[0].split("-")
                    if len(f_split) == 2:
                        bandList.append(pjoin(root, f))
                        bandNums.append(os.path.splitext(f)[0].split("-")[1])

        if not bandList:
            raise Exception(
                "Could not find any geotifs in '{0}'".format(self.group_path)
            )
        return bandList, bandNums

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

    def resample_bands(
        self,
        bandList,
        resample_spatialRes,
        resample_option,
        load=False,
        save=True,
        odir=None,
    ):
        """
        Resample bands (listed in resample_bands) to the specified
        spatial resolution using the specified resampling option
        such as bilinear interpolation, nearest neighbour, cubic
        etc. Bands have the option of being saved

        Parameters
        ----------
        bandList : list
            A list of paths to bands that will be resampled

        resample_spatialRes : float > 0
            The resampled spatial resolution

        resample_option : <enum 'Resampling'> class
            The Resampling.<> algorithm, e.g.
            Resampling.nearest
            Resampling.bilinear
            Resampling.cubic
            Resampling.cubic_spline
            e.t.c

        load : bool
            if True
                The resampled bands are returned as a numpy.ndarray
                having dimensions of
                [nRows, nCols] or [nBands, nRows, nCols]
                
            if False
                The resampled bands are not returned

        save : bool
            if True
                geotiff is saved into either
                (a) odir (if specified), or;
                (b) directory of ref_band (if odir=None)

            if False
                geotiff is not saved

        odir : str or None
            The directory where the resampled geotiff is
            saved. if odir=None then ref_band directory
            is used.

        Returns
        -------
        resmpl_ofiles : list or None
            A list of successful resampled bands that were saved.
            resmpl_ofiles = None if save = False
            
        spectral_cube : numpy.ndarray or None
            A numpy array containing the resampled bands
            spectral_cube = None if load = False

        metad : dict or None
            A dictionary containing (modified) rasterio metadata
            of spectral_cube.
            metad = None if load = False

        Raises
        ------
        Exception
            * resample_spatialRes <= 0
            * if save=True and odir does not exist.

        Notes
        -----
         1) if load is False, then spectral_cube & metad are None
         2) if load is True, then spectral_cube = numpy.ndarray
            and metad = dict()
         3) if save is False, then resmpl_ofiles = None
         4) if save is True and resmpl_ofiles = []
            then resampling was not successful for any listed bands

        """
        if resample_spatialRes <= 0:
            raise Exception("\nresample_spatialRes must be a float > 0")

        metad = None
        data_type = None
        spectral_cube = None
        resmpl_ofiles = None

        nBands = len(bandList)

        if save:
            resmpl_ofiles = []
            if odir:
                out_dir = odir
                self.check_path_exists(out_dir)  # Exception if not exist
            else:
                out_dir = os.path.dirname(ref_band)

        if load:
            spectral_cube = []

        # iterate through the list of bands to be resampled
        for z in range(0, nBands):

            with rasterio.open(bandList[z], "r") as src_ds:
                data_type = np.dtype(src_ds.dtypes[0])
                dwnscale_factor = resample_spatialRes / src_ds.transform.a
                nRows = int(src_ds.height / dwnscale_factor)
                nCols = int(src_ds.width / dwnscale_factor)

                res_transform = src_ds.transform * src_ds.transform.scale(
                    dwnscale_factor, dwnscale_factor
                )

                metad = src_ds.meta.copy()
                metad["width"] = nCols
                metad["height"] = nRows
                metad["transform"] = res_transform

                # do stuff to get ref_ds.transform, ref_ds.crs
                # ref_ds.height, ref_ds.width
                resmpl_band = np.zeros(
                    [nRows, nCols], order="C", dtype=data_type
                )

                reproject(
                    source=rasterio.band(src_ds, 1),
                    destination=resmpl_band,
                    src_transform=src_ds.transform,
                    src_crs=src_ds.crs,
                    dst_transform=res_transform,
                    dst_crs=src_ds.crs,
                    resampling=resample_option,
                )

                metad["band_{0}".format(z + 1)] = self.get_basename(
                    bandList[z]
                )

                if load:
                    spectral_cube.append(np.copy(resmpl_band))

                if save:
                    # save geotif
                    resmpl_tif = pjoin(
                        out_dir,
                        self.get_resample_bandName(
                            bandList[z], resample_spatialRes
                        ),
                    )

                    with rasterio.open(resmpl_tif, "w", **metad) as dst:
                        dst.write(resmpl_band, 1)

                    if os.path.exists(resmpl_tif):
                        # geotiff was successfully created.
                        resmpl_ofiles.append(resmpl_tif)

            # end-with src_ds
        # end-for z

        if load:
            # update nBands in metadata
            metad["count"] = nBands

            if nBands == 1:
                # return a 2-dimensional numpy.ndarray
                spectral_cube = np.copy(spectral_cube[0])

            else:
                # return a 3-dimensional numpy.ndarray
                spectral_cube = np.array(
                    spectral_cube, order="C", dtype=data_type
                )
        else:
            metad = None

        # if load is False:
        #     spectral_cube & metad are None
        # if save is False:
        #     resmpl_ofiles = None
        # if (save is True) and (len(resmpl_ofiles) == 0):
        #     resampling was not successful for any listed bands

        return resmpl_ofiles, spectral_cube, metad

    def load_fmask(self, fmask_file):
        """
        Loads the fmask.img file as a numpy.ndarray

        Parameters
        ----------
        fmask_file : str
            filename of the fmask.img
        """
        with rasterio.open(fmask_file, "r") as ds:
            fmask_img = ds.read(1)
        return fmask_img

    def load_bands(self, band_list, apply_scaling):
        """
        load the bands in band_list into a 3D array.
        
        Parameters
        ----------
        band_list : list of paths
            paths of bands to load. Note these bands
            must be the same spatial resolution

        apply_scaling : bool
            if True, then the scale_factor is applied

        Returns
        -------
        spectral_cube : numpy.ndarray
            multi-band array with dimensions of
            [nBands, nRows, nCols]
        """
        nBands = len(band_list)
        spectral_cube = None
        nodata_val = None
        meta_dict = None
        data_type = np.float32

        for z in range(0, nBands):
            with rasterio.open(band_list[z], "r") as ds:
                if z == 0:
                    if not apply_scaling:
                        data_type = np.dtype(ds.dtypes[0])

                    spectral_cube = np.zeros(
                        [nBands, ds.height, ds.width],
                        order="C",
                        dtype=data_type,
                    )
                    meta_dict = ds.meta.copy()

                band = np.array(ds.read(1), order="C", dtype=data_type)

                nodata_val = float(ds.meta["nodata"])

                if apply_scaling:
                    band[
                        (band > 0) & (band <= self.scale_factor)
                    ] /= self.scale_factor

                band[band > self.scale_factor] = nodata_val
                band[band <= 0] = nodata_val

                spectral_cube[z, :, :] = band
                meta_dict["band_{0}".format(z + 1)] = self.get_basename(
                    band_list[z]
                )

        meta_dict["count"] = nBands
        return spectral_cube, meta_dict

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

        if (self.sensor == "sentinel_2") or (self.sensor == "landsat_8"):
            ix_red = self.band_ids.index("4")
            ix_grn = self.band_ids.index("3")
            ix_blu = self.band_ids.index("2")

        if (self.sensor == "landsat_5") or (self.sensor == "landsat_7"):
            ix_red = self.band_ids.index("3")
            ix_grn = self.band_ids.index("2")
            ix_blu = self.band_ids.index("1")

        if self.sensor == "worldview_2":
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
            resmpl_tifs, refl_im, rio_meta = self.resample_bands(
                rgb_bandList,
                ql_spatialRes,
                Resampling.nearest,
                load=True,
                save=False,
                odir=None,
            )
        else:
            refl_im, rio_meta = self.load_bands(rgb_bandList, False)

        # resample fmask
        resmpl_tifs, fmask_im, fmask_meta = self.resample_bands(
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
        ax = self.display_img(rgb_im)
        mc = ROI_Selector(ax=ax)
        mc.interative()
        plt.show()

        # write a shapefile
        mc.verts_to_shp(metadata=rgb_meta, shp_file=shp_file)

        # close the ROI_Selector
        mc = None

    def load_mask_from_shp(self, shp_file, ref_ds):
        """
        Load a mask containing geometries from a shapefile,
        using a reference dataset

        Parameters
        ----------
        shp_file : str
            shapefile containing a polygon

        ref_ds : dataset
            A rasterio dataset of the reference band
            that fmask will be resampled

        Returns
        -------
        mask_im : numpy.ndarray
            mask image

        Notes
        -----
        1) Pixels outside of the polygon are assigned
           as nodata in the mask
        """
        with fiona.open(shp_file, "r") as shp:
            shapes = [feature["geometry"] for feature in shp]

        # pixels outside polygon in roi_mask = nodata value
        mask_im, mask_transform = rasterio.mask.mask(
            dataset=ref_ds, shapes=shapes
        )
        if mask_im.ndim == 3:
            mask_im = np.array(mask_im[0, :, :], order="K", copy=True)

        return mask_im

    def resample_band_to_ds(self, image_file, ref_ds, resample_option):
        """
        Resample fmask to a reference band given
        by the rasterio dataset (ref_ds)

        Parameters
        ----------
        image_file : str
            filename of image to resample

        ref_ds : dataset
            A rasterio dataset of the reference band
            that fmask will be resampled

        resample_option : <enum 'Resampling'> class
            The Resampling.<> algorithm, e.g.
            Resampling.nearest
            Resampling.bilinear
            Resampling.cubic
            Resampling.cubic_spline
            e.t.c

        Returns
        -------
        image : numpy.ndarray
            2-Dimensional numpy array
        """
        with rasterio.open(image_file, "r") as src_ds:
            image = np.zeros(
                [ref_ds.height, ref_ds.width],
                order="C",
                dtype=np.dtype(src_ds.dtypes[0]),
            )

            reproject(
                source=rasterio.band(src_ds, 1),
                destination=image,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                dst_transform=ref_ds.transform,
                dst_crs=ref_ds.crs,
                resampling=resample_option,
            )
        return image

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
            The path where the deglinted geotiff bands
            are saved.

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
            os.makedirs(odir)

        # Check that the input vis bands exist
        self.check_bandNum_exist(self.band_ids, vis_band_ids)
        self.check_bandNum_exist(self.band_ids, [nir_band_id])

        ix_nir = self.band_ids.index(nir_band_id)
        nir_bandPath = self.bandList[ix_nir]

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
            vis_bandPath = self.bandList[ix_vis]

            # ------------------------------ #
            #        load visible band       #
            # ------------------------------ #
            with rasterio.open(vis_bandPath, "r") as vis_ds:
                vis_im = vis_ds.read(1)

                # get metadata and load NIR array
                kwargs = vis_ds.meta.copy()
                nRows = vis_ds.height
                nCols = vis_ds.width
                nodata = kwargs["nodata"]
                vis_transform = vis_ds.transform
                spatialRes = int(abs(kwargs["transform"].a))

                deglint_otif = pjoin(
                    odir,
                    "{0}-deglint-{1}m.tif".format(
                        self.get_basename(vis_bandPath), spatialRes,
                    ),
                )
                # ------------------------------ #
                #  Resample and load fmask_file  #
                # ------------------------------ #
                fmask = self.resample_band_to_ds(
                    self.fmask_file, vis_ds, Resampling.mode
                )

                # ------------------------------ #
                #       Resample NIR band        #
                # ------------------------------ #
                if (nir_nRows != nRows) or (nir_nCols != nCols):
                    # resample the NIR band to match the VIS band
                    nir_im = self.resample_band_to_ds(
                        nir_bandPath, vis_ds, Resampling.bilinear
                    )

            # ------------------------------ #
            #       Sunglint Correction      #
            # ------------------------------ #
            # copy band
            deglint_band = np.array(vis_im, order="K", copy=True)

            waterIx = np.where(fmask == waterVal)

            # 1. deglint water pixels
            deglint_band[waterIx] = vis_im[waterIx] - nir_im[waterIx]

            # 2. write geotiff
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
            os.makedirs(odir)

        # initiate plot if specified
        if plot:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # Check that the input vis bands exist
        self.check_bandNum_exist(self.band_ids, vis_band_ids)
        self.check_bandNum_exist(self.band_ids, [nir_band_id])

        ix_nir = self.band_ids.index(nir_band_id)
        nir_bandPath = self.bandList[ix_nir]

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
            vis_bandPath = self.bandList[ix_vis]

            # ------------------------------ #
            #        load visible band       #
            # ------------------------------ #
            with rasterio.open(vis_bandPath, "r") as vis_ds:
                vis_im = vis_ds.read(1)

                # get metadata and load NIR array
                kwargs = vis_ds.meta.copy()
                nRows = vis_ds.height
                nCols = vis_ds.width
                nodata = kwargs["nodata"]
                vis_transform = vis_ds.transform
                spatialRes = int(abs(kwargs["transform"].a))

                deglint_otif = pjoin(
                    odir,
                    "{0}-deglint-{1}m.tif".format(
                        self.get_basename(vis_bandPath), spatialRes,
                    ),
                )
                # ------------------------------ #
                #  Resample and load fmask_file  #
                # ------------------------------ #
                fmask = self.resample_band_to_ds(
                    self.fmask_file, vis_ds, Resampling.mode
                )

                # ------------------------------ #
                #    Load shapefile as a mask    #
                # ------------------------------ #
                roi_mask = self.load_mask_from_shp(roi_shpfile, vis_ds)
                roi_mask[(fmask != waterVal) & (vis_im == nodata)] = nodata

                # ------------------------------ #
                #       Resample NIR band        #
                # ------------------------------ #
                if (nir_nRows != nRows) or (nir_nCols != nCols):
                    # resample the NIR band to match the VIS band
                    nir_im = self.resample_band_to_ds(
                        nir_bandPath, vis_ds, Resampling.bilinear
                    )

            # ------------------------------ #
            #       Sunglint Correction      #
            # ------------------------------ #
            # copy band
            deglint_band = np.array(vis_im, order="K", copy=True)

            # 1. Find minimum NIR in the roi polygon
            roi_valIx = np.where(roi_mask != nodata)
            valid_NIR = nir_im[roi_valIx]
            minRefl_NIR = valid_NIR.min()

            waterIx = np.where(fmask == waterVal)

            # 2. Get correlations between current band and NIR
            x_vals = valid_NIR
            y_vals = vis_im[roi_valIx]
            slope, y_inter, r_val, p_val, std_err = stats.linregress(
                x=x_vals, y=y_vals
            )

            # 3. deglint water pixels
            deglint_band[waterIx] = vis_im[waterIx] - slope * (
                nir_im[waterIx] - minRefl_NIR
            )

            # 4. write geotiff
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
