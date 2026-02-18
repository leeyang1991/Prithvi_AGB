from __global__ import *

from lytools import *
from shapely.geometry import box
from shapely.geometry import mapping
from rasterio.mask import mask
from pprint import pprint

class RasterIO_Func_Extend(RasterIO_Func):
    # todo: add to lytools

    def __init__(self):
        super().__init__()
        pass

    def clip_tif_by_bounds(self,in_tif,out_tif,bounds):


        with rasterio.open(in_tif) as src:
            minx, miny, maxx, maxy = bounds
            input_bounds = box(minx, miny, maxx, maxy)
            geo_df = gpd.GeoDataFrame({'geometry': input_bounds}, index=[0],
                                      crs=src.crs)

            # Get the geometry coordinates in the format rasterio mask function expects
            # which is a list of GeoJSON-like objects
            geoms = [mapping(g) for g in geo_df.geometry.values]

            # Clip the raster
            out_image, out_transform = mask(dataset=src, shapes=geoms, crop=True)

            # Update metadata for the output file
            # src.pro
            profile = src.profile
            profile.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src.crs
            })

            with rasterio.open(out_tif, "w", **profile) as dest:
                dest.write(out_image)

        pass

    def get_tif_bounds_(self,fpath):
        # rename to get_tif_bounds
        with rasterio.open(fpath) as src:
            profile = src.profile
        # pprint(profile)
        # exit()
        crs = profile['crs']
        originX = profile['transform'][2]
        originY = profile['transform'][5]
        pixelWidth = profile['transform'][0]
        pixelHeight = profile['transform'][4]
        # endX = originX + array.shape[1] * pixelWidth
        endX = originX + profile['width'] * pixelWidth
        endY = originY + profile['height'] * pixelHeight

        minx, miny, maxx, maxy = originX, endY, endX, originY
        return minx, miny, maxx, maxy

    def clip_tif_by_tif(self,in_tif,out_tif,clip_tif):
        bounds = self.get_tif_bounds_(clip_tif)
        self.clip_tif_by_bounds(in_tif,out_tif,bounds)