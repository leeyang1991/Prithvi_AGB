from __init__ import *
from __global__ import *
this_script_root = join(data_root,'GEDI')
import rasterio

class Preprocess_GEDI:
    def __init__(self):
        self.region = 'AZ'
        self.data_dir = this_script_root
        pass

    def run(self):
        self.crop()
        # self.padding_224()
        pass

    @Decorator.shutup_gdal
    def crop(self):
        import HLS
        GEDI_global_fpath = join(self.data_dir,'tif','gedi_2019-2023.tif')
        HLS_1km_template = join(HLS.Preprocess_HLS().data_dir,'1.5_resample_30m_to_1km',f'{self.region}_6_bands_resample_1km.tif')
        outdir = join(self.data_dir,'crop')
        T.mkdir(outdir,force=True)

        GEDI_out_fpath = join(outdir,f'{self.region}.tif')
        array, originX, originY, pixelWidth, pixelHeight, projection_wkt = self.raster2array(HLS_1km_template)
        left = originX
        right = originX + (array.shape[1]) * pixelWidth
        bottom = originY + (array.shape[0]) * pixelHeight
        top = originY
        crop_window = (left, bottom, right, top)
        dstSRS = global_gedi_WKT()
        res = global_res_gedi
        self.clip_tif_using_coordinates(GEDI_global_fpath, GEDI_out_fpath, crop_window,dstSRS,res)
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