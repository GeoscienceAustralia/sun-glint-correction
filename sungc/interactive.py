#!/usr/bin/env python3

import sys
import fiona
import rasterio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from typing import Optional, Union
from shapely import geometry
from matplotlib import patches
from PIL import Image, ImageDraw

"""
## --------------------------------------------------- ##
## --------------------------------------------------- ##
##     CODE THAT ALLOWS THE USER TO INTERACTIVELY      ##
##            SELECT A ROI FROM IMAGERY                ##
## --------------------------------------------------- ##
## --------------------------------------------------- ##
"""


class RoiSelector(object):
    """
    An interactive polygon editor.

    Parameters
    ----------
    ax: matplotlib.axes.Axes

    poly_xy : list of (float, float)
        List of (x, y) coordinates used as vertices of the polygon.

    max_ds : float
        Max pixel distance to count as a vertex hit.

    Key-bindings
    ------------
    't' : toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

    'd' : delete the vertex under point

    'i' : insert a vertex at point.  You must be within max_ds of the
          line connecting two existing vertices
    """

    def __init__(
        self,
        ax: matplotlib.axes.Axes,
        poly_xy: Optional[Union[list, tuple, None]] = None,
        max_ds: Optional[Union[int, float] = 10],
    ):
        self.showverts = True
        self.max_ds = max_ds
        self.ax = ax
        self.poly_xy = poly_xy

        if not isinstance(ax, matplotlib.axes.Axes):
            raise Exception("ax in RoiSelector must be matplotlib.axes.Axes")

        if not isinstance(max_ds, (float, int)):
            raise Exception("max_ds in RoiSelector must be int or float")

        if not isinstance(poly_xy, (list, tuple, type(None))):
            raise Exception("poly_xy in RoiSelector must be a list, tuple or None")

        if isinstance(poly_xy, (list, tuple)):
            # poly_xy contains the vertices of a polygon. The smallest
            # polygon possible is a triangle, hence nVertices >= 3
            if len(poly_xy) < 3:
                raise Exception("input ploy_xy in RoiSelector must have >= 3 vertices")

        if max_ds <= 0:
            raise Exception("max_ds in RoiSelector must be > 0")

    def interative(self):

        # get the default vertices
        if not self.poly_xy:
            self.poly_xy = self.default_vertices(self.ax)

        # add polygon to axis
        self.poly = patches.Polygon(
            self.poly_xy, animated=True, fc="y", ec="none", alpha=0.4
        )
        self.ax.add_patch(self.poly)

        self.ax.set_clip_on(False)
        self.ax.set_title(
            "Select homogeneous water region with varying sunglint."
            "\nClick and drag a point to move it; "
            "'i' to insert; 'd' to delete.\n"
            "Close figure when done."
        )

        x, y = zip(*self.poly.xy)
        self.line = plt.Line2D(
            x, y, color="none", marker="o", mfc="r", alpha=0.2, animated=True
        )
        self._update_line()
        self.ax.add_line(self.line)

        self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas = self.poly.figure.canvas
        canvas.mpl_connect("draw_event", self.draw_callback)
        canvas.mpl_connect("button_press_event", self.button_press_callback)
        canvas.mpl_connect("button_release_event", self.button_release_callback)
        canvas.mpl_connect("key_press_event", self.key_press_callback)
        canvas.mpl_connect("motion_notify_event", self.motion_notify_callback)
        self.canvas = canvas

    def default_vertices(self, ax):
        """
        Default to rectangle that has a quarter-width/height border.
        """
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        w = np.diff(xlims)
        h = np.diff(ylims)
        x1, x2 = xlims + w // 4 * np.array([1, -1])
        y1, y2 = ylims + h // 4 * np.array([1, -1])

        return ((x1, y1), (x1, y2), (x2, y2), (x2, y1))

    def verts_to_shp(self, metadata, shp_file):
        """
        Save the user-selected polygon vertices to
        an ascii file.

        Parameters
        ----------
        metadata : dict()
            A rasterio dictionary containing image metadata

        shp_file : str
            shapefile that will contain the vertices of a polygon
        """
        poly_coords = []
        for xy_vt in self.verts:
            x, y = rasterio.transform.xy(
                transform=metadata["transform"],
                cols=xy_vt[0],
                rows=xy_vt[1],
                offset="center",
            )
            poly_coords.append((x, y))

        # Define a polygon feature geometry with one attribute
        schema = {"geometry": "Polygon", "properties": {"id": "str"}}

        # Write a new Shapefile
        with fiona.open(shp_file, "w", "ESRI Shapefile", schema) as c:
            c.write(
                {
                    "geometry": geometry.mapping(geometry.Polygon(poly_coords)),
                    "properties": {"id": "deep-water-polygon"},
                }
            )

    def verts_from_shp(self, shp_file):
        """
        Load polygon vertices from a shapefile

        Parameters
        ----------
        shp_file: str
            shapefile that contains a polygon
        """
        with fiona.open(shp_file, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]

        print(shapes)
        sys.exit()

    def get_mask(self, shape):
        """Return image mask given by mask creator"""
        h, w = shape
        # The original code imported nxutils from matplotlib.
        # However, this didn't work because it was deprecated.
        # PIL was used instead.

        polygon = []
        for v in range(0, len(self.verts)):
            # print(self.verts[v])
            polygon.append((self.verts[v][0], self.verts[v][1]))

        img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        return np.array(img, order="C", dtype=np.uint8)

    def poly_changed(self, poly):
        "this method is called whenever the polygon object is called"
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        # Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def dist(self, x, y):
        """
        Return the euclidean distance between two points.

        The original code can be accessed from
        https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mlab.py
        """
        d = x - y
        return (np.dot(d, d)) ** 0.5

    def dist_point_to_segment(self, p, s0, s1):
        """
        Get the distance of a point to a segment.
          *p*, *s0*, *s1* are *xy* sequences

        This algorithm was from taken from
        http://geomalgorithms.com/a02-_lines.html

        The original code can be accessed from
        https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mlab.py
        """
        p = np.array(p, order="C", copy=False, dtype=np.float64)
        s0 = np.array(s0, order="C", copy=False, dtype=np.float64)
        s1 = np.array(s1, order="C", copy=False, dtype=np.float64)
        v = s1 - s0
        w = p - s0

        c1 = np.dot(w, v)
        if c1 <= 0:
            return self.dist(p, s0)

        c2 = np.dot(v, v)
        if c2 <= c1:
            return self.dist(p, s1)

        b = c1 / c2
        pb = s0 + b * v
        return self.dist(p, pb)

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)

    def button_press_callback(self, event):
        "whenever a mouse button is pressed"
        ignore = not self.showverts or event.inaxes is None or event.button != 1
        if ignore:
            return
        self._ind = self.get_ind_under_cursor(event)

    def button_release_callback(self, event):
        "whenever a mouse button is released"
        ignore = not self.showverts or event.button != 1
        if ignore:
            return
        self._ind = None

    def key_press_callback(self, event):
        "whenever a key is pressed"
        if not event.inaxes:
            return

        if event.key == "t":
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None

        elif event.key == "d":
            ind = self.get_ind_under_cursor(event)
            if ind is None:
                return
            if ind == 0 or ind == self.last_vert_ind:
                print("Cannot delete root node")
                return
            self.poly.xy = [tup for i, tup in enumerate(self.poly.xy) if i != ind]
            self._update_line()

        elif event.key == "i":
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # cursor coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = self.dist_point_to_segment(p, s0, s1)
                if d <= self.max_ds:
                    self.poly.xy = np.array(
                        list(self.poly.xy[: i + 1])
                        + [(event.xdata, event.ydata)]
                        + list(self.poly.xy[i + 1 :])
                    )
                    self._update_line()
                    break

        self.canvas.draw()

    def motion_notify_callback(self, event):
        "on mouse movement"
        ignore = (
            not self.showverts
            or event.inaxes is None
            or event.button != 1
            or self._ind is None
        )
        if ignore:
            return
        x, y = event.xdata, event.ydata

        if self._ind == 0 or self._ind == self.last_vert_ind:
            self.poly.xy[0] = x, y
            self.poly.xy[self.last_vert_ind] = x, y
        else:
            self.poly.xy[self._ind] = x, y

        self._update_line()
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def _update_line(self):
        # save verts because polygon gets deleted when figure is closed
        self.verts = self.poly.xy
        self.last_vert_ind = len(self.poly.xy) - 1
        self.line.set_data(zip(*self.poly.xy))

    def get_ind_under_cursor(self, event):
        "get the index of the vertex under cursor if within max_ds tolerance"
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if d[ind] >= self.max_ds:
            ind = None
        return ind
