import matplotlib.pyplot as plt

from __init__ import *

from __global__ import *
import rasterio
import geopandas as gp
import earthaccess
import ee
import urllib3

from lytools import *
T = Tools()
this_script_root = join(data_root,'additional_index')

class HLS_Vegetation_Index:
    def __init__(self):
        # if
        # self.resolution = '1km'
        self.resolution = '30m'
        self.data_dir = join(this_script_root,'HLS_Vegetation_Index')
        pass

    def run(self):
        # self.cal_ndvi()
        # self.cal_NDWI()
        self.cal_NBR()
        pass

    def cal_ndvi(self):
        T.mkdir(self.data_dir,force=True)
        import HLS
        fdir = join(HLS.Preprocess_HLS().data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands')
        if self.resolution == '1km':
            fpath = join(fdir,'B2-B7_1km.tif')
            outf = join(self.data_dir,'ndvi_1km.tif')
        elif self.resolution == '30m':
            fpath = join(fdir,'B2-B7.tif')
            outf = join(self.data_dir,'ndvi_30m.tif')
        else:
            raise
        with rasterio.open(fpath) as src:
            profile = src.profile
            profile['count'] = 1

            data = src.read()
            red_band = data[2]
            nir_band = data[3]
            ndvi = (nir_band - red_band) / (nir_band + red_band)
            # print(np.shape(ndvi))
            # ndvi[ndvi<=0] = np.nan
            # ndvi[ndvi>1] = np.nan
            ndvi = np.array([ndvi])
            # exit()
            with rasterio.open(outf, "w", **profile) as dst:
                dst.write(ndvi)
        print('done')

    def cal_NDWI(self):
        T.mkdir(self.data_dir, force=True)
        import HLS
        fdir = join(HLS.Preprocess_HLS().data_dir, 'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands')

        if self.resolution == '1km':
            fpath = join(fdir,'B2-B7_1km.tif')
            outf_ndwi = join(self.data_dir, 'ndwi_1km.tif')
            outf_mndwi = join(self.data_dir, 'mndwi_1km.tif')
        elif self.resolution == '30m':
            fpath = join(fdir,'B2-B7.tif')
            outf_ndwi = join(self.data_dir, 'ndwi_30m.tif')
            outf_mndwi = join(self.data_dir, 'mndwi_30m.tif')
        else:
            raise

        with rasterio.open(fpath) as src:
            profile = src.profile
            profile['count'] = 1

            data = src.read()
            green_band = data[1]
            swir1_band = data[4]
            nir_band = data[3]
            ndwi = (green_band - nir_band) / (green_band + nir_band)
            MNDWI = (green_band - swir1_band) / (green_band + swir1_band)
            # print(np.shape(ndvi))
            # ndwi[ndwi < -1] = np.nan
            # ndwi[ndwi > 1] = np.nan
            ndwi = np.array([ndwi])

            # MNDWI[MNDWI < -1] = np.nan
            # MNDWI[MNDWI > 1] = np.nan
            MNDWI = np.array([MNDWI])

            with rasterio.open(outf_ndwi, "w", **profile) as dst:
                dst.write(ndwi)
            with rasterio.open(outf_mndwi, "w", **profile) as dst:
                dst.write(MNDWI)
        print('done')
        pass

    def cal_NBR(self):
        T.mkdir(self.data_dir, force=True)
        import HLS
        fdir = join(HLS.Preprocess_HLS().data_dir, 'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands')

        if self.resolution == '1km':
            fpath = join(fdir,'B2-B7_1km.tif')
            outf_nbr = join(self.data_dir, 'nbr_1km.tif')
        elif self.resolution == '30m':
            fpath = join(fdir,'B2-B7.tif')
            outf_nbr = join(self.data_dir, 'nbr_30m.tif')
        else:
            raise

        with rasterio.open(fpath) as src:
            profile = src.profile
            profile['count'] = 1

            data = src.read()
            green_band = data[1]
            swir1_band = data[4]
            swir2_band = data[5]
            nir_band = data[3]
            nbr = (nir_band - swir2_band) / (nir_band + swir2_band)
            # print(np.shape(ndvi))
            # nbr[nbr < -1] = np.nan
            # nbr[nbr > 1] = np.nan
            nbr = np.array([nbr])

            with rasterio.open(outf_nbr, "w", **profile) as dst:
                dst.write(nbr)
        print('done')
        pass


class DEM_1km:

    def __init__(self):
        self.data_dir = join(this_script_root,'DEM')

    def run(self):
        # self.download_images()
        # self.unzip()
        # self.merge()
        # self.reproj()
        self.crop()
        pass

    def download_images(self):
        outdir = join(self.data_dir,'zips')
        T.mkdir(outdir,force=True)
        ee.Initialize(project='lyfq-263413')

        resolution = global_res_gedi
        band_name = 'elevation'
        product_name = 'USGS/SRTMGL1_003'
        Image_band = ee.Image(product_name)
        region_list = self.rectangle(rect=[-180, 90, 180, -90],block_res=15)
        flag = 1
        for region in tqdm(region_list):
            flag += 1
            outf_name = join(outdir, f'{flag}.zip')
            if isfile(outf_name):
                return

            # print(region)
            exportOptions = {
                'scale': resolution,
                'region': region,
            }
            url = Image_band.getDownloadURL(exportOptions)
            try:
                self.download_i(url, outf_name)
            except:
                print('download error', outf_name)

    def rectangle(self,rect=(-180, 90, 180, -90),block_res=90):
        rect_list = []
        lon_start = rect[0]
        lat_start = rect[3]
        lon_end = rect[2]
        lat_end = rect[1]
        for lon in np.arange(lon_start, lon_end, block_res):
            for lat in np.arange(lat_start, lat_end, block_res):
                rect_i = [lon, lat, lon + block_res, lat + block_res]
                # print(rect_i)
                rect_i_new = [rect_i[0], rect_i[3], rect_i[2], rect_i[1]]
                rect_i_new = [float(i) for i in rect_i_new]
                # exit()
                rect_list.append(rect_i_new)
        # print(rect_list)
        # print('len(rect_list)', len(rect_list))
        rect_list_obj = []
        for rect_i in rect_list:
            # print(rect_i)
            rect_i_obj = ee.Geometry.Rectangle(rect_i[0], rect_i[1], rect_i[2], rect_i[3])
            # rect_i_obj = ee.Geometry.Rectangle(rect_i[0], rect_i[1], rect_i[2], rect_i[3])
            rect_list_obj.append(rect_i_obj)
        return rect_list_obj


    def download_i(self,url,outf):
        # try:
        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)
        body = r.read()
        with open(outf, 'wb') as f:
            f.write(body)

    def unzip(self):
        fdir = join(self.data_dir,r'zips')
        outdir = join(self.data_dir,r'unzip')
        T.mk_dir(outdir,force=True)
        self._unzip(fdir,outdir)
        pass

    def _unzip(self, zipfolder, outdir):
        # zipfolder = join(self.datadir,'zips')
        # outdir = join(self.datadir,'unzip')
        T.mkdir(outdir)
        for f in tqdm(T.listdir(zipfolder)):
            outdir_i = join(outdir, f.replace('.zip', ''))
            T.mkdir(outdir_i)
            fpath = join(zipfolder, f)
            # print(fpath)
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(outdir_i)
            zip_ref.close()

    @Decorator.shutup_gdal
    def merge(self):
        fdir = join(self.data_dir,r'unzip')
        outdir = join(self.data_dir,r'merge')
        T.mk_dir(outdir,force=True)
        fpath_list = []

        for date in tqdm(T.listdir(fdir)):
            fdir_i = join(fdir,date)
            for f in T.listdir(fdir_i):
                if not f.endswith('.tif'):
                    continue
                fpath = join(fdir_i,f)
                fpath_list.append(fpath)
        srcSRS = DIC_and_TIF().gen_srs_from_wkt(global_wkt_84())
        # srcSRS = DIC_and_TIF().gen_srs_from_wkt(self.wkt_sin())
        outf = join(outdir,f'DEM_1km.tif')
        gdal.Warp(outf,fpath_list,srcSRS=srcSRS, outputType=gdal.GDT_Int32)

        pass

    @Decorator.shutup_gdal
    def reproj(self):
        fpath = join(self.data_dir,'merge/DEM_1km.tif')
        outpath = join(self.data_dir,'merge/DEM_1km_reproj6933.tif')
        dstSRS = global_gedi_WKT()
        srcSRS = global_wkt_84()
        ToRaster().resample_reproj(fpath,outpath,global_res_gedi,srcSRS=srcSRS, dstSRS=dstSRS)
        pass

    @Decorator.shutup_gdal
    def crop(self):
        import HLS
        import GEDI
        fpath = join(self.data_dir,'merge/DEM_1km_reproj6933.tif')
        outpath = join(self.data_dir,'merge/DEM_1km_reproj6933_crop.tif')
        template_fpath = join(HLS.Preprocess_HLS().data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7_1km.tif')

        array, originX, originY, pixelWidth, pixelHeight, projection_wkt = GEDI.Preprocess_GEDI().raster2array(template_fpath)
        left = originX
        right = originX + (array.shape[1]) * pixelWidth
        bottom = originY + (array.shape[0]) * pixelHeight
        top = originY
        crop_window = (left, bottom, right, top)

        GEDI.Preprocess_GEDI().clip_tif_using_coordinates(fpath,outpath,crop_window,global_gedi_WKT(),global_res_gedi)

class DEM_30m:

    def __init__(self):
        self.data_dir = join(this_script_root,'DEM_30m')

    def run(self):
        # self.download_images()
        # self.unzip()
        # self.merge()
        # self.reproj()
        self.crop()
        pass

    def download_images(self):
        outdir = join(self.data_dir,'zips')
        T.mkdir(outdir,force=True)
        ee.Initialize(project='lyfq-263413')

        resolution = global_res_hls
        band_name = 'elevation'
        product_name = 'USGS/SRTMGL1_003'
        Image_band = ee.Image(product_name)
        # region_list = self.rectangle(rect=[-180, 90, 180, -90],block_res=15)
        region_list = self.rectangle(rect=[-115, 39, -110, 36],block_res=1)
        flag = 1
        for region in tqdm(region_list):
            flag += 1
            outf_name = join(outdir, f'{flag}.zip')
            if isfile(outf_name):
                return

            # print(region)
            exportOptions = {
                'scale': resolution,
                'region': region,
            }
            url = Image_band.getDownloadURL(exportOptions)
            try:
                self.download_i(url, outf_name)
            except:
                print('download error', outf_name)

    def rectangle(self,rect=(-180, 90, 180, -90),block_res=90):
        rect_list = []
        lon_start = rect[0]
        lat_start = rect[3]
        lon_end = rect[2]
        lat_end = rect[1]
        for lon in np.arange(lon_start, lon_end, block_res):
            for lat in np.arange(lat_start, lat_end, block_res):
                rect_i = [lon, lat, lon + block_res, lat + block_res]
                # print(rect_i)
                rect_i_new = [rect_i[0], rect_i[3], rect_i[2], rect_i[1]]
                rect_i_new = [float(i) for i in rect_i_new]
                # exit()
                rect_list.append(rect_i_new)
        # print(rect_list)
        # print('len(rect_list)', len(rect_list))
        rect_list_obj = []
        for rect_i in rect_list:
            # print(rect_i)
            rect_i_obj = ee.Geometry.Rectangle(rect_i[0], rect_i[1], rect_i[2], rect_i[3])
            # rect_i_obj = ee.Geometry.Rectangle(rect_i[0], rect_i[1], rect_i[2], rect_i[3])
            rect_list_obj.append(rect_i_obj)
        return rect_list_obj


    def download_i(self,url,outf):
        # try:
        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)
        body = r.read()
        with open(outf, 'wb') as f:
            f.write(body)

    def unzip(self):
        fdir = join(self.data_dir,r'zips')
        outdir = join(self.data_dir,r'unzip')
        T.mk_dir(outdir,force=True)
        self._unzip(fdir,outdir)
        pass

    def _unzip(self, zipfolder, outdir):
        # zipfolder = join(self.datadir,'zips')
        # outdir = join(self.datadir,'unzip')
        T.mkdir(outdir)
        for f in tqdm(T.listdir(zipfolder)):
            outdir_i = join(outdir, f.replace('.zip', ''))
            T.mkdir(outdir_i)
            fpath = join(zipfolder, f)
            # print(fpath)
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(outdir_i)
            zip_ref.close()

    @Decorator.shutup_gdal
    def merge(self):
        fdir = join(self.data_dir,r'unzip')
        outdir = join(self.data_dir,r'merge')
        T.mk_dir(outdir,force=True)
        fpath_list = []

        for date in tqdm(T.listdir(fdir)):
            fdir_i = join(fdir,date)
            for f in T.listdir(fdir_i):
                if not f.endswith('.tif'):
                    continue
                fpath = join(fdir_i,f)
                fpath_list.append(fpath)
        srcSRS = DIC_and_TIF().gen_srs_from_wkt(global_wkt_84())
        # srcSRS = DIC_and_TIF().gen_srs_from_wkt(self.wkt_sin())
        outf = join(outdir,f'DEM_30m.tif')
        gdal.Warp(outf,fpath_list,srcSRS=srcSRS, outputType=gdal.GDT_Int32)

        pass

    @Decorator.shutup_gdal
    def reproj(self):
        fpath = join(self.data_dir,'merge/DEM_30m.tif')
        outpath = join(self.data_dir,'merge/DEM_30m_reproj6933.tif')
        dstSRS = global_gedi_WKT()
        srcSRS = global_wkt_84()
        ToRaster().resample_reproj(fpath,outpath,global_res_hls,srcSRS=srcSRS, dstSRS=dstSRS)
        pass

    @Decorator.shutup_gdal
    def crop(self):
        import HLS
        import GEDI
        fpath = join(self.data_dir,'merge/DEM_30m_reproj6933.tif')
        outpath = join(self.data_dir,'merge/DEM_30m_reproj6933_crop.tif')
        template_fpath = join(HLS.Preprocess_HLS().data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7.tif')

        array, originX, originY, pixelWidth, pixelHeight, projection_wkt = GEDI.Preprocess_GEDI().raster2array(template_fpath)
        left = originX
        right = originX + (array.shape[1]) * pixelWidth
        bottom = originY + (array.shape[0]) * pixelHeight
        top = originY
        crop_window = (left, bottom, right, top)

        GEDI.Preprocess_GEDI().clip_tif_using_coordinates(fpath,outpath,crop_window,global_gedi_WKT(),global_res_hls)

def main():
    HLS_Vegetation_Index().run()
    # DEM_1km().run()
    # DEM_30m().run()
    pass

if __name__ == '__main__':
    main()