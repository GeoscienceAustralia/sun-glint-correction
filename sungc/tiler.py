#!/usr/bin/env python3

import numpy as np


def generate_tiles(samples, lines, xtile=None, ytile=None):
    """
    Generates a list of tile indices for a 2D array. Taken from
    https://github.com/GeoscienceAustralia/wagl/blob/water-atcor/wagl/tiling.py

    Parameters
    ----------
    samples : int
        An integer expressing the total number of samples (columns)
        in an array.

    lines : int
        An integer expressing the total number of lines (rows)
        in an array.

    xtile : int or None
        (Optional) The desired size of the tile in the x-direction.
        Default is all samples

    ytile : int or None
        (Optional) The desired size of the tile in the y-direction.
        Default is min(100, lines) lines.

    Returns
    -------
        Each tuple in the generator contains
        ((ystart,yend),(xstart,xend)).

    Examples
    --------
        >>> from wagl.tiling import generate_tiles
        >>> tiles = generate_tiles(8624, 7567, xtile=1000, ytile=400)
        >>> for tile in tiles:
        >>>     # A rasterio dataset
        >>>     subset = rio_ds.read([4, 3, 2], window=tile)
        >>>     # Or simply move the tile window across an array
        >>>     subset = array[tile]  # 2D
        >>>     subset = array[:,tile[0],tile[1]]  # 3D
    """

    def create_tiles(samples, lines, xstart, ystart):
        """
        Creates a generator object for the tiles.
        """
        for ystep in ystart:
            if ystep + ytile < lines:
                yend = ystep + ytile
            else:
                yend = lines
            for xstep in xstart:
                if xstep + xtile < samples:
                    xend = xstep + xtile
                else:
                    xend = samples
                yield (slice(ystep, yend), slice(xstep, xend))

    # check for default or out of bounds
    if xtile is None or xtile < 0:
        xtile = samples
    if ytile is None or ytile < 0:
        ytile = min(100, lines)

    xstart = np.arange(0, samples, xtile)
    ystart = np.arange(0, lines, ytile)

    tiles = create_tiles(samples, lines, xstart, ystart)

    return tiles
