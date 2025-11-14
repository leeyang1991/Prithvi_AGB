import matplotlib.pyplot as plt

from __init__ import *
from __global__ import *
this_script_root = join(data_root,'GEDI')
import rasterio

class Preprocess_GEDI:
    def __init__(self):
        self.data_dir = this_script_root
        pass

    def run(self):
        # self.crop()
        self.padding_224()
        pass

    def get_WKT(self):
        projection_wkt = '''
        PROJCRS["WGS 84 / NSIDC EASE-Grid 2.0 Global",
            BASEGEOGCRS["WGS 84",
                ENSEMBLE["World Geodetic System 1984 ensemble",
                    MEMBER["World Geodetic System 1984 (Transit)"],
                    MEMBER["World Geodetic System 1984 (G730)"],
                    MEMBER["World Geodetic System 1984 (G873)"],
                    MEMBER["World Geodetic System 1984 (G1150)"],
                    MEMBER["World Geodetic System 1984 (G1674)"],
                    MEMBER["World Geodetic System 1984 (G1762)"],
                    ELLIPSOID["WGS 84",6378137,298.257223563,
                        LENGTHUNIT["metre",1]],
                    ENSEMBLEACCURACY[2.0]],
                PRIMEM["Greenwich",0,
                    ANGLEUNIT["degree",0.0174532925199433]],
                ID["EPSG",4326]],
            CONVERSION["US NSIDC EASE-Grid 2.0 Global",
                METHOD["Lambert Cylindrical Equal Area",
                    ID["EPSG",9835]],
                PARAMETER["Latitude of 1st standard parallel",30,
                    ANGLEUNIT["degree",0.0174532925199433],
                    ID["EPSG",8823]],
                PARAMETER["Longitude of natural origin",0,
                    ANGLEUNIT["degree",0.0174532925199433],
                    ID["EPSG",8802]],
                PARAMETER["False easting",0,
                    LENGTHUNIT["metre",1],
                    ID["EPSG",8806]],
                PARAMETER["False northing",0,
                    LENGTHUNIT["metre",1],
                    ID["EPSG",8807]]],
            CS[Cartesian,2],
                AXIS["easting (X)",east,
                    ORDER[1],
                    LENGTHUNIT["metre",1]],
                AXIS["northing (Y)",north,
                    ORDER[2],
                    LENGTHUNIT["metre",1]],
            USAGE[
                SCOPE["Environmental science - used as basis for EASE grid."],
                AREA["World between 86°S and 86°N."],
                BBOX[-86,-180,86,180]],
            ID["EPSG",6933]]'''

        return projection_wkt

    @Decorator.shutup_gdal
    def crop(self):
        import HLS
        HLS_tif_dir = join(HLS.Preprocess_HLS().data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands')
        GEDI_fpath = join(self.data_dir,'tif','gedi_2019-2023.tif')
        fpath = join(HLS_tif_dir,'B2-B7_1km.tif')
        array, originX, originY, pixelWidth, pixelHeight, projection_wkt = self.raster2array(fpath)
        left = originX
        right = originX + (array.shape[1]) * pixelWidth
        bottom = originY + (array.shape[0]) * pixelHeight
        top = originY
        crop_window = (left, bottom, right, top)
        dstSRS = self.get_WKT()
        res = 1000.89502334966744
        self.clip_tif_using_coordinates(GEDI_fpath, join(self.data_dir,'tif','gedi_2019-2023_clipped.tif'), crop_window,dstSRS,res)
        print('done')

    def clip_tif_using_coordinates(self,in_tif,out_tif,crop_window,dstSRS,res):
        # todo: add to lytools
        # crop_window: (upper_left_x, upper_left_y, lower_right_x, lower_right_y)
        # gdal.Translate(out_tif, in_tif, projWin=crop_window)
        xmin, ymin, xmax, ymax = crop_window
        gdal.Warp(out_tif, in_tif, format="GTiff", outputBounds=[xmin, ymin, xmax, ymax],srcSRS=dstSRS,dstSRS=dstSRS,xRes=res, yRes=res)
        # gdal.Warp(out_tif, in_tif, format="GTiff", outputBounds=[xmin, ymin, xmax, ymax],xRes=res, yRes=res)
        pass

    def padding_224(self):
        fpath = join(self.data_dir,'tif/gedi_2019-2023_clipped.tif')
        outf = join(self.data_dir,'tif/gedi_2019-2023_clipped_224.tif')
        from rasterio.transform import Affine

        with rasterio.open(fpath) as src:
            profile = src.profile
            data = src.read()

        H, W = data.shape[1:]
        pad_h = (224 - H % 224) % 224
        pad_w = (224 - W % 224) % 224
        data_pad = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=np.nan)

        # 更新元数据
        profile.update(height=data_pad.shape[1], width=data_pad.shape[2])

        with rasterio.open(outf, "w", **profile) as dst:
            dst.write(data_pad)
        pass

    def raster2array(self, rasterfn):
        '''
        create array from raster
        Agrs:
            rasterfn: tiff file path
        Returns:
            array: tiff data, an 2D array
        '''
        raster = gdal.Open(rasterfn)
        projection_wkt = raster.GetProjection()
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        array = np.asarray(array)
        del raster
        return array, originX, originY, pixelWidth, pixelHeight,projection_wkt

def main():
    Preprocess_GEDI().run()
    pass

if __name__ == '__main__':
    main()