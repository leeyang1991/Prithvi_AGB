import copy

import affine
import math

import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import load_entry_point
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling

import geopandas as gpd
from __init__ import *

from __global__ import *
import rasterio
import geopandas as gp
import earthaccess
import ee
import urllib3
from pathlib import Path

# ee.Authenticate()
# exit()
from lytools import *
T = Tools()
# exit()
this_script_root = join(data_root,'HLS')

class Download:

    def __init__(self):
        # self.data_dir = '/data/home/wenzhang/Yang/HLS/'
        self.data_dir = this_script_root
        self.get_account_passwd()
        self.geojson_fpath = join(data_root,'global_tiles_HLS','nm.geojson')

        pass

    def run(self):
        # self.kml_to_shp()
        # self.gen_urls()
        self.download()
        # self.check_download()
        # self.move_tile_to_different_folder()
        # self.delete_empty_folders()
        pass

    def kml_to_shp(self):
        # download from:
        # https://hls.gsfc.nasa.gov/wp-content/uploads/2016/03/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml
        # website: https://hls.gsfc.nasa.gov/products-description/tiling-system/
        import geopandas as gpd
        import shapely
        from shapely import wkt
        kml_fpath = join(data_root, 'global_tiles_HLS',
                         'S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml')
        outf = join(data_root, 'global_tiles_HLS', 'HLS_tiles.shp')
        print(isfile(kml_fpath))
        gdf = gpd.read_file(kml_fpath, driver="KML")
        gdf.drop(columns=['Description'], inplace=True)

        geometry_list = []
        for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
            GEOMETRYCOLLECTION = row.geometry
            GEOMETRYCOLLECTION_2d = shapely.force_2d(GEOMETRYCOLLECTION)
            GEOMETRYCOLLECTION_2d = str(GEOMETRYCOLLECTION_2d)

            geom = wkt.loads(GEOMETRYCOLLECTION_2d)
            polygons = [g for g in geom.geoms if g.geom_type == "Polygon"]
            poly = polygons[0]
            geometry_list.append(poly)
        gdf["geometry"] = geometry_list
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf.set_crs("EPSG:4326")
        gdf.to_file(outf)
        print('done')


    def get_tiles_from_geojson(self,fpath):
        # print(isfile(fpath))
        gdf = gp.read_file(fpath)
        Name_list = gdf['Name'].to_list()
        Name_list = [f'T{name}' for name in Name_list]
        return Name_list


    def gen_urls_test(self):
        # outdir = join(self.data_dir, 'url_list')
        # T.mkdir(outdir, force=True)
        # outf = join(outdir, 'url_list.txt')
        # if isfile(outf):
        #     return
        earthaccess.login(persist=True)
        # earthaccess.download()
        field = gp.read_file('az_test.geojson')
        bbox = tuple(list(field.total_bounds))
        # print(bbox)
        # exit()
        temporal = ("2019-01-01T00:00:00", "2023-12-31T23:59:59")

        results = earthaccess.search_data(
            short_name='Global_Veg_Greenness_GIMMS_3G_2187',
            temporal=temporal,
            count=5
        )
        print(results)
        exit()
        # filelist = earthaccess.download(results, local_path=outdir)

        hls_results_urls = [granule.data_links() for granule in results]
        # print('len(hls_results_urls)', len(hls_results_urls))
        # fw = open(outf,'w')
        # for url_list in hls_results_urls:
        #     for url in url_list:
        #         fw.write(url + '\n')
        # fw.close()
        pass

    def gen_urls(self):
        geojson_fpath = self.geojson_fpath
        outdir = join(self.data_dir, 'url_list')
        T.mkdir(outdir, force=True)
        geojson_name = Path(geojson_fpath).name
        outf = join(outdir, geojson_fpath.replace('.geojson', '.txt'))
        if isfile(outf):
            return
        earthaccess.login(persist=True)
        # earthaccess.download()
        field = gp.read_file(geojson_fpath)
        bbox = tuple(list(field.total_bounds))
        tiles_list = self.get_tiles_from_geojson(geojson_fpath)
        # print(tiles_list)
        # exit()
        temporal = ("2019-01-01T00:00:00", "2023-12-31T23:59:59")

        results = earthaccess.search_data(
            short_name=['HLSL30', 'HLSS30'],
            bounding_box=bbox,
            temporal=temporal,
            # count=5
        )
        # print(results)
        # filelist = earthaccess.download(results, local_path=outdir)

        hls_results_urls = [granule.data_links() for granule in results]
        print('len(hls_results_urls)', len(hls_results_urls))
        fw = open(outf,'w')
        for url_list in hls_results_urls:
            for url in url_list:
                url_tile = url.split('/')[-1].split('.')[2]
                if not url_tile in tiles_list:
                    continue
                fw.write(url + '\n')
        fw.close()
        pass

    def download(self):
        outdir = join(self.data_dir, 'Download')
    #
        selected_bands = [
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
            'Fmask'
        ]
        hls_results_urls = []
        geojson_name = Path(self.geojson_fpath).name
        with open(join(self.data_dir, 'url_list', geojson_name.replace('.geojson', '.txt')), 'r') as f:
            for line in f:
                hls_results_urls.append(line.strip())
        session = requests.Session()
        session.auth = requests.auth.HTTPBasicAuth(self.username, self.password)
        params_list = []
        for url in hls_results_urls:
            # print(url)
            band = url.split('.')[-2]
            if not band in selected_bands:
                continue
            tile = url.split('/')[-1].split('.')[2]
            outdir_i = join(outdir,tile)
            params_list.append([url,session,outdir_i])
            # print(url)
            # outdir_i = join(outdir,url.split('/')[-2])
            # T.mkdir(outdir_i)
            # outf = join(outdir_i,url.split('/')[-1])
            # self.kernel_download([url,session,outdir])
        MULTIPROCESS(self.kernel_download,params_list).run(process=10, process_or_thread='t')

    def kernel_download(self,params):
        url,session,outdir = params
        outdir_i = join(outdir,url.split('/')[-2])
        try:
            T.mkdir(outdir_i,force=True)
        except:
            pass
        outf = join(outdir_i,url.split('/')[-1])
        if isfile(outf):
            return
        try:
            self.download_i(outf,session,url)
        except Exception as e:
            print('error')
            print(e)
            print('--------')

        fail_time = 0
        while 1:
            ok = self.check_download_single_file(outf)
            if ok and fail_time > 0:
                print(f'download successful after {fail_time} times trials')
                fail_time = 0
                break
            else:
                fail_time += 1
                if fail_time > 10:
                    print('download failed after 10 times trials')
                os.remove(outf)
                sleep(10)
                try:
                    self.download_i(outf, session, url)
                except Exception as e:
                    print(f'error times {fail_time}: {url}')
                    print(e)
                    print('--------')

    def download_i(self,outf,session,url):
        fw = open(outf, 'wb')

        r = session.get(url, stream=True)
        for chunk in r.iter_content(chunk_size=8192):
            fw.write(chunk)
        fw.close()
        pass

    def check_download(self):
        tiles = self.get_tiles_from_geojson(self.geojson_fpath)
        print(tiles)
        fdir = join(self.data_dir,'Download')
        params_list = []
        for i,tile in enumerate(tiles):
            for folder in T.listdir(join(fdir,tile)):
                params = [fdir,tile,folder]
                params_list.append(params)
                # self.kernel_check_download(params)
        MULTIPROCESS(self.kernel_check_download,params_list).run(process=10, process_or_thread='t')
        pass

    def kernel_check_download(self,params):
        fdir,tile,folder = params
        for f in T.listdir(join(fdir, tile, folder)):
            fpath = join(fdir, tile, folder, f)
            ok = self.check_download_single_file(fpath)
            if not ok:
                print(fpath)
                os.remove(fpath)
        pass

    def check_download_single_file(self,fpath):
        try:
            with rasterio.open(fpath) as src:
                profile = src.profile
                height = profile['height']
                width = profile['width']
                data = src.read(window=((height - 1, height), (width - 1, width)))
                # data = src.read(window=((0, 1), (0, 1)))
                return True
        except:
            return False

    def move_tile_to_different_folder(self):
        # fdir = join(self.data_dir,'Download')
        fdir = '/data/home/wenzhang/Yang/Transformer_Learn/data/HLS/Download'
        for folder in tqdm(T.listdir(fdir)):
            for f in T.listdir(join(fdir,folder)):
                fpath = join(fdir,folder,f)
                tile = f.split('.')[2]
                outdir_i = join(fdir,tile,folder)
                T.mkdir(outdir_i,force=True)
                outf = join(outdir_i,f)
                os.rename(fpath,outf)
                # print(outf)
                # exit()
        pass

    def delete_empty_folders(self):

        # fdir = join(self.data_dir,'Download')
        fdir = fdir = '/data/home/wenzhang/Yang/Transformer_Learn/data/HLS/Download'
        for folder in tqdm(T.listdir(fdir)):
            if len(T.listdir(join(fdir,folder)))==0:
                os.rmdir(join(fdir,folder))
        pass

    def get_account_passwd(self):
        passwd_fpath = 'passwd.txt'
        with open(passwd_fpath, 'r') as f:
            lines = f.readlines()
        account = lines[0].strip()
        passwd = lines[1].strip()
        self.username = account
        self.password = passwd


class Preprocess_HLS:
    def __init__(self):
        self.data_dir = this_script_root
        self.conf()
        pass

    def run(self):
        # self.gen_image_shp()
        self.quality_control_long_term_mean()
        # self.mosaic_merge_bands_tifs()
        # self.re_proj()
        # self.mosaic_utah()
        # self.plot_time_series()
        # self.get_tif_template()
        # self.resample_to_1km()
        # self.padding_224()

        pass

    def conf(self):
        self.bands_list = [
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
        ]
        self.year_list = ['2019', '2020', '2021', '2022', '2023']
        self.sat_list = ['L30', 'S30']
        # self.tile_list = ['T12STG','T12STH','T12SUG','T12SUH','T12SVG','T12SVH']
        sat = self.sat_list[0]
        band = self.bands_list[0]
        year = self.year_list[0]
        block_dict = {}
        # for tile in self.tile_list:
        #     block_list = T.listdir(join(self.data_dir,'reproj_qa_concatenate','/'.join([sat,band,year,tile])))
        #     block_dict[tile] = block_list
        self.block_dict = block_dict
        pass

    def re_proj(self):
        fdir = join(self.data_dir,'mosaic_merge_bands_tifs')
        outdir = join(self.data_dir,'mosaic_merge_bands_tifs_reproj')
        T.mkdir(outdir,force=True)
        dst_crs = global_gedi_WKT()
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            RasterIO_Func().reproject_tif(fpath,outf,dst_crs,dst_crs_res=30)

    def mosaic_utah(self):
        tile_list = ['T12STG', 'T12STH', 'T12SUG', 'T12SUH', 'T12SVG', 'T12SVH']
        fdir = join(self.data_dir,'mosaic_merge_bands_tifs_reproj')
        outdir = join(self.data_dir,'mosaic_utah')
        T.mkdir(outdir,force=True)
        outf = join(outdir,'mosaic_utah_reproj.tif')
        flist = [join(fdir,f'{tile}.tif') for tile in tile_list]
        RasterIO_Func().mosaic_tifs(flist,outf)
        pass


    def get_WKT(self,fpath):
        raster = gdal.Open(fpath)
        projection_wkt = raster.GetProjection()
        return projection_wkt

    def quality_control_long_term_mean(self):
        geojson_fpath = Download().geojson_fpath
        tile_list = Download().get_tiles_from_geojson(geojson_fpath)
        # print(tile_list)
        # exit()
        outdir = join(self.data_dir,'quality_control_long_term_mean')
        T.mkdir(outdir,force=True)
        njob = 32
        memory_allocate = 0.5 # in GB, 6nGB, njob = tot mem/6nGB
        qa_filter_list = self.get_qa_filter_list()
        band_list = self.bands_list
        fdir = join(self.data_dir,'Download')
        params_list = []
        # for tile in T.listdir(fdir):
        for tile in tile_list:
            dtype = np.int16
            # get QA 3d array
            flist_qa = []
            for folder in T.listdir(join(fdir,tile)):
                qa_fpath = join(fdir,tile, folder, f'{folder}.Fmask.tif')
                flist_qa.append(qa_fpath)
            # flist_qa = flist_qa[:10]
            Tif_loader_qa = Tif_loader(flist_qa,memory_allocate,dtype=dtype,mute=True)

            # get each band 3d array
            for band in band_list:
                band_fpath_list = []
                for folder in T.listdir(join(fdir, tile)):
                    band_fpath = join(fdir, tile, folder, f'{folder}.{band}.tif')
                    band_fpath_list.append(band_fpath)
                # band_fpath_list = band_fpath_list[:10]
                Tif_loader_band = Tif_loader(band_fpath_list,memory_allocate,dtype=dtype,mute=True)
                for idx in Tif_loader_band.block_index_list:
                    params = [Tif_loader_qa,idx,qa_filter_list,Tif_loader_band,outdir,tile,band]
                    params_list.append(params)
                    # self.kernel_quality_control_long_term_mean(params)
        MULTIPROCESS(self.kernel_quality_control_long_term_mean,params_list).run(process=njob)

    def kernel_quality_control_long_term_mean(self,params):
        Tif_loader_qa,idx,qa_filter_list,Tif_loader_band,outdir,tile,band = params
        iter_length = Tif_loader_qa.iter_length
        digit = math.log(iter_length,10) + 1
        digit = int(digit)
        outdir_i = join(outdir, tile, band)
        outf = join(outdir_i, f'{tile}.{band}.{idx:0{digit}d}.tif')
        # print(outf)
        if isfile(outf):
            return

        qa_patch_concat, qa_profile = Tif_loader_qa.array_iterator_index(idx)
        qa_filter_mask_list = []
        for qa_patch_concat_i in qa_patch_concat:
            qa_filter_mask = self.gen_qa_mask_array(qa_patch_concat_i, qa_filter_list)
            qa_filter_mask = qa_filter_mask.astype(bool)
            qa_filter_mask_list.append(qa_filter_mask)
        qa_filter_mask_concat = np.stack(qa_filter_mask_list, axis=0)
        band_patch_concat, band_profile = Tif_loader_band.array_iterator_index(idx)
        # band_patch_concat[~qa_filter_mask_concat] = -9999
        # band_patch_concat = np.array(band_patch_concat, dtype=np.float32)
        # band_patch_concat[band_patch_concat == -9999] = np.nan
        # band_patch_concat_average = np.nanmean(band_patch_concat, axis=0)
        # print(qa_filter_mask_concat.shape)
        band_patch_concat_average = np.mean(band_patch_concat, axis=0,where=qa_filter_mask_concat)
        # delta = band_patch_concat_average1 - band_patch_concat_average
        # plt.imshow(band_patch_concat_average)
        # plt.show()
        # plt.imshow(band_patch_concat_average1)
        # plt.show()
        # plt.imshow(delta)
        # plt.show()
        # pause()

        try:
            T.mkdir(outdir_i, force=True)
        except:
            pass
        RasterIO_Func().write_tif(band_patch_concat_average, outf, band_profile)


    def mosaic_merge_bands_tifs(self):
        fdir = join(self.data_dir,'quality_control_long_term_mean')
        outdir = join(self.data_dir,'mosaic_merge_bands_tifs1')
        T.mkdir(outdir,force=True)
        flag = 0
        total_flag = len(T.listdir(fdir))
        for tile in T.listdir(fdir):
            mosaic_list = []
            band_name_list = []
            out_profile = ''
            flag += 1
            for band in tqdm(T.listdir(join(fdir,tile)),desc=f'{flag}/{total_flag} {tile}'):
                array_list = []
                profile_list = []
                for f in T.listdir(join(fdir,tile,band)):
                    fpath = join(fdir,tile,band,f)
                    array,profile = RasterIO_Func().read_tif(fpath)
                    array_list.append(array)
                    profile_list.append(profile)
                mosaic,out_profile = RasterIO_Func().mosaic_arrays(array_list,profile_list)
                mosaic = mosaic.squeeze()
                mosaic_list.append(mosaic)
                band_name_list.append(band)
            mosaic_list = np.array(mosaic_list)
            outf = join(outdir,f'{tile}.tif')
            RasterIO_Func().write_tif_multi_bands(mosaic_list, outf, out_profile, band_name_list)

            pass
        pass

    def quality_control1(self):
        memory_allocate = 0.1 # in GB
        qa_filter_list = self.get_qa_filter_list()
        band_list = self.bands_list
        fdir = join(self.data_dir,'Download')

        for tile in T.listdir(fdir):
            dtype = np.int16
            # get QA 3d array
            flist_qa = []
            for folder in T.listdir(join(fdir,tile)):
                qa_fpath = join(fdir,tile, folder, f'{folder}.Fmask.tif')
                flist_qa.append(qa_fpath)
            Tif_loader_qa = Tif_loader(flist_qa,memory_allocate,dtype=dtype)
            qa_band_array_dict = {}
            for idx in tqdm(Tif_loader_qa.block_index_list,desc='loading qa'):
                qa_patch_concat, qa_profile = Tif_loader_qa.array_iterator_index(idx)
                qa_filter_mask_list = []
                for qa_patch_concat_i in qa_patch_concat:
                    qa_filter_mask = self.gen_qa_mask_array(qa_patch_concat_i,qa_filter_list)
                    # print(qa_filter_mask.shape)
                    qa_filter_mask_list.append(qa_filter_mask)
                qa_filter_mask_concat = np.stack(qa_filter_mask_list,axis=0)
                # print(qa_filter_mask_concat.shape)
                qa_band_array_dict[idx] = qa_filter_mask_concat

            # get each band 3d array
            for band in band_list:
                band_fpath_list = []
                for folder in T.listdir(join(fdir, tile)):
                    band_fpath = join(fdir, tile, folder, f'{folder}.{band}.tif')
                    band_fpath_list.append(band_fpath)
                Tif_loader_band = Tif_loader(band_fpath_list,memory_allocate,dtype=dtype)
                for idx in Tif_loader_band.block_index_list:
                    qa_patch_concat = qa_band_array_dict[idx]
                    band_patch_concat, band_profile = Tif_loader_band.array_iterator_index(idx)
                    print(qa_patch_concat.shape)
                    print(band_patch_concat.shape)
                    band_patch_concat_mask_list = []
                    for i in tqdm(range(qa_patch_concat.shape[0])):
                        band_patch_mask = band_patch_concat[i][~qa_patch_concat[i]]
                        band_patch_concat_mask_list.append(band_patch_mask)
                        if i == 0:
                            print(band_patch_mask.shape)
                    band_patch_concat_mask = np.stack(band_patch_concat_mask_list,axis=0)
                    print(band_patch_concat_mask.shape)
                    pause()


                exit()


    def get_qa_filter_list(self):
        # print('need water body!!!!')
        # raise 'need water body!!!!'
        '''
        see:https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf#page=17.08
        fmask value:
        clean pixel values for No water/snow_ice/cloud/cloud_shadow/Adjacent_to_cloud:
        [0,64,128,192]
        bit num|mask name        |bit value|mask description
        7-6    |aerosol level    |11       |High aerosol
        7-6    |aerosol level    |10       |Moderate aerosol
        7-6    |aerosol level    |01       |Low aerosol
        7-6    |aerosol level    |00       |Climatology aerosol
        5      |Water            |1        |Yes
        5      |Water            |0        |No
        4      |Snow/ice         |1        |Yes
        4      |Snow/ice         |0        |No
        3      |Cloud shadow     |1        |Yes
        3      |Cloud shadow     |0        |No
        2      |Adjacent to cloud|1        |Yes
        2      |Adjacent to cloud|0        |No
        1      |Cloud            |1        |Yes
        1      |Cloud            |0        |No
        0      |Cirrus           |NA       |NA
        '''
        # qa_filter_list = [
        #     0b00000000,
        #     0b10000000,
        #     0b11000000,
        #     0b01000000,
        #     0b00100000,
        #     0b10100000,
        #     0b11100000,
        #     0b01100000,
        # ]
        # qa_filter_list.sort()
        # print(qa_filter_list)
        qa_filter_list = []
        for bit in range(pow(2, 3)):
            # print(bit*2**5)
            qa_filter_list.append(bit * pow(2, 5))
        return qa_filter_list

    def gen_qa_mask_array(self,qa_array,qa_filter_list):
        arr_init = np.ones(qa_array.shape, dtype=np.uint8)
        arr_init_copy_qa_list = []
        for qa_code in qa_filter_list:
            arr_init_copy = copy.copy(qa_code)
            arr_init_copy_qa = arr_init_copy & qa_array
            arr_init_copy_qa_list.append(arr_init_copy_qa)

        arr_filter_init = np.zeros(qa_array.shape, dtype=np.uint8)
        for arr_init_copy_qa in arr_init_copy_qa_list:
            arr_filter = qa_array == arr_init_copy_qa
            arr_filter_init += arr_filter
        arr_filter_init[arr_filter_init>0] = 1

        return arr_filter_init

    def gen_image_shp(self):
        from shapely.geometry import Polygon
        import geopandas as gpd
        outdir = join(self.data_dir,'image_shp')
        T.mkdir(outdir)
        fdir = join(self.data_dir,'Download')
        tile_list = []
        shp_flist = []
        for tile in T.listdir(fdir):
            for folder in T.listdir(join(fdir,tile)):
                for f in T.listdir(join(fdir,tile,folder)):
                    fpath = join(fdir,tile,folder,f)
                    # print(fpath)
                    # print(isfile(fpath))
                    # exit()
                    # array, originX, originY, pixelWidth, pixelHeight, _ = self.raster2array(fpath)
                    array,profile = RasterIO_Func().read_tif(fpath)
                    crs = profile['crs']
                    array = array.squeeze()
                    # pprint(profile)
                    originX = profile['transform'][2]
                    originY = profile['transform'][5]
                    pixelWidth = profile['transform'][0]
                    pixelHeight = profile['transform'][4]
                    endX = originX+array.shape[1]*pixelWidth
                    endY = originY+array.shape[0]*pixelHeight
                    ll_point = (originX, originY)
                    lr_point = (endX, originY)
                    ur_point = (endX, endY)
                    ul_point = (originX, endY)
                    # print(ll_point, lr_point, ur_point, ul_point)
                    # exit()
                    polygon_geom = [Polygon([ll_point, lr_point, ur_point, ul_point])]
                    outf = join(outdir,f'{tile}.shp')
                    crs = crs.to_string()
                    # print(crs)
                    # exit()
                    # gpd.GeoDataFrame(geometry=[polygon_geom],crs=crs).to_file(outf)
                    polygon = gpd.GeoDataFrame(crs=crs, geometry=polygon_geom)
                    polygon['tile'] = [tile]
                    polygon.to_file(outf)
                    tile_list.append(tile)
                    shp_flist.append(outf)
                    break
                break

        gdfs = [gpd.read_file(f).to_crs("EPSG:4326") for f in shp_flist]
        gdf_merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        gdf_merged.to_file(join(outdir,'tiles_merge.shp'))
        for f in T.listdir(outdir):
            for tile in tile_list:
                if tile in f:
                    os.remove(join(outdir,f))

    def get_tif_template(self):
        tile_list = self.tile_list
        fdir_reproj = join(self.data_dir, 'reproj')
        tile_template_dict = {}
        for tile in tile_list:
            template_tif = ''
            for folder in T.listdir(fdir_reproj):
                tile_ = folder.split('.')[2]
                if tile == tile_:
                    template_tif = join(fdir_reproj, folder, f'{folder}.B02.tif')
                    break
            if len(template_tif) == 0:
                raise
            tile_template_dict[tile] = template_tif
        return tile_template_dict


    def plot_time_series(self):
        fdir = join(self.data_dir,'reproj_qa_concatenate')
        sat = 'L30'
        band = 'B02'
        year = '2019'
        tile = 'T12STG'
        fdir_i = join(fdir,'/'.join([sat,band,year,tile]))
        for f in T.listdir(fdir_i):
            fpath = join(fdir_i,f)
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                if T.is_all_nan(vals):
                    continue
                print(vals)
                plt.plot(vals)
                plt.scatter(list(range(len(vals))),vals)
                plt.title(pix)
                plt.show()
                pause()

        pass

    def resample_to_1km(self):
        import GEDI
        # fpath = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7.tif')
        fpath = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/mosaic10.tif')
        # outf = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7_1km.tif')
        outf = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/mosaic10_1km.tif')
        SRS = GEDI.Preprocess_GEDI().get_WKT()
        res = 1000.89502334966744
        ToRaster().resample_reproj(fpath,outf,res,SRS,SRS)
        pass

    def padding_224(self):
        fpath = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7_1km.tif')
        outf = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7_1km_224.tif')

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

    def array2raster(self, newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array, projection_wkt,ndv=-999999):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = longitude_start
        originY = latitude_start
        # open geotiff
        driver = gdal.GetDriverByName('GTiff')
        if os.path.exists(newRasterfn):
            os.remove(newRasterfn)
        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
        # outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
        # ndv = 255
        # Add Color Table
        # outRaster.GetRasterBand(1).SetRasterColorTable(ct)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        # Write Date to geotiff
        outband = outRaster.GetRasterBand(1)

        outband.SetNoDataValue(ndv)
        outband.WriteArray(array)
        outRasterSRS = osr.SpatialReference()
        # outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(projection_wkt)
        # Close Geotiff
        outband.FlushCache()
        del outRaster

    def steps(self):
        '''
        download 500g

        qa 500g
        concatenate 500g*2
        aggragate 14g


        reproj
        mosaic
        merge-bands
        '''
        pass

class Download_From_GEE_1km:

    def __init__(self):
        self.data_dir = join(this_script_root,'Download_from_GEE_1km')

        self.collection = 'NASA/HLS/HLSL30/v002' # derived from LandSAT
        # --------------------------------------------------------------------------------
        # self.collection = 'NASA/HLS/HLSS30/v002'  # derived from Sentinel

        self.product = self.collection.split('/')[-2]
        pass

    def run(self):
        date_range_list = self.gen_date_list()
        # pprint(date_range_list)
        # exit()
        ee.Initialize(project='lyfq-263413')
        outdir = join(self.data_dir,'zips',self.product)
        T.mk_dir(outdir,force=True)

        param_list = []
        for startDate,endDate in date_range_list:
            param = outdir,startDate,endDate
            param_list.append(param)
            # self.download_images(param)
        MULTIPROCESS(self.download_images,param_list).run(process=20,process_or_thread='t')

    def download_images(self,param):
        outdir,startDate,endDate = param
        # print(site)
        res = global_res_gedi
        outdir_i = join(outdir)
        T.mk_dir(outdir_i)

        Collection = ee.ImageCollection(self.collection)
        Collection = Collection.filterDate(startDate, endDate)

        info_dict = Collection.getInfo()
        # pprint(info_dict)
        # print(len(info_dict['features']))
        # exit()
        ids = info_dict['features']
        # for i in tqdm(ids):
        for i in ids:
            dict_i = eval(str(i))
            # pprint.pprint(dict_i['id'])
            # exit()
            outf_name = dict_i['id'].split('/')[-1] + '.zip'
            out_path = join(outdir_i, outf_name)
            if isfile(out_path):
                continue
            Image = ee.Image(dict_i['id'])
            # Image_product = Image.select('total_precipitation')
            Image_product = Image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Fmask'])
            # print(Image_product);exit()
            # region = [-111, 32.2, -110, 32.6]# left, bottom, right,
            # region = [-180, -90, 180, 90]  # left, bottom, right,
            exportOptions = {
                'scale': res,
                'maxPixels': 1e13,
                # 'region': region,
                # 'fileNamePrefix': 'exampleExport',
                # 'description': 'imageToAssetExample',
            }
            url = Image_product.getDownloadURL(exportOptions)
            # print(url)

            try:
                self.download_i(url, out_path)
            except:
                print('download error', out_path)
                continue
        pass
    def download_i(self,url,outf):
        # try:
        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)
        body = r.read()
        with open(outf, 'wb') as f:
            f.write(body)

    def gen_date_list(self,start_date = '2019-01-01',end_date = '2023-12-31'):

        days_count = T.count_days_of_two_dates(start_date, end_date)
        # print(days_count)
        # exit()
        date_list = []
        base_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        # for i in range(int(days_count/3)+1):
        for i in np.arange(0,days_count+2,1):
            # print(i)
            date = base_date + datetime.timedelta(days=int(i))
            date_list.append(date.strftime('%Y-%m-%d'))
        # pprint(date_list)
        # exit()
        # every two days
        # date_range_list = []
        # for i in range(len(date_list)-1):
        #     if i%2==0:
        #         date_range_list.append([date_list[i],date_list[i+1]])
        # every one days
        date_range_list = []
        for i in range(len(date_list) - 1):
            date_range_list.append([date_list[i], date_list[i]])
        return date_range_list

class Tif_loader:

    def __init__(self,flist,memory_allocate,dtype=None,nodata=None,mute=False):
        self.flist = flist
        self.memory_allocate = memory_allocate
        self.profile = self.get_image_profiles(flist[0])
        if nodata==None:
            pass
        else:
            self.profile.update(nodata=nodata)
        # pprint(profile)
        self.h = self.profile['height']
        self.w = self.profile['width']
        if dtype == None:
            self.dtype = self.profile['dtype']
        else:
            self.dtype = dtype
        self.profile.update(dtype=self.dtype)
        # pprint(self.profile)
        self.available_rows = self.get_available_rows(self.memory_allocate, len(flist), self.h, self.w, dtype=self.dtype)
        self.iter_length = math.ceil(self.h / self.available_rows)
        self.block_index_list = list(range(math.ceil(self.h / self.available_rows)))
        if not mute:
            print('input file size:', f'h:{self.h},w:{self.w}')
            print('input file count:', len(self.flist))
            print('output block size:', f'h:{self.available_rows},w:{self.w}')
            print('output block count:', self.iter_length)
            print('------------------')
        pass

    def array_iterator(self):
        idx = 0
        for row in range(0, self.h, self.available_rows):
            patch_concat_list = []
            for fpath in self.flist:
                with rasterio.open(fpath) as src:
                    patch = src.read(
                        window=((row, row + self.available_rows),(0, self.w))
                    )
                    patch_concat_list.append(patch)
            patch_concat = np.concatenate(patch_concat_list, axis=0)
            patch_concat_list = []

            window = Window(col_off=0, row_off=self.available_rows * idx, width=self.w, height=self.available_rows)
            new_transform = rasterio.windows.transform(window, src.transform)

            profile_new = self.profile.copy()
            profile_new['height'] = self.available_rows
            profile_new['transform'] = new_transform
            transform = profile_new['transform']
            idx += 1
            yield patch_concat, profile_new

    def array_iterator_index(self,idx):

        row_list = list(range(0, self.h, self.available_rows))
        row = row_list[idx]
        patch_concat_list = []
        for fpath in self.flist:
            with rasterio.open(fpath) as src:
                patch = src.read(
                    window=((row, row + self.available_rows),(0, self.w))
                )
                patch_concat_list.append(patch)
        patch_concat = np.concatenate(patch_concat_list, axis=0)
        # print(patch_concat.shape)
        # exit()
        patch_concat_list = []

        window = Window(col_off=0, row_off=self.available_rows*idx, width=self.w,height=patch_concat.shape[1])
        new_transform = rasterio.windows.transform(window, src.transform)

        profile_new = self.profile.copy()
        profile_new['height'] = patch_concat.shape[1]
        profile_new['transform'] = new_transform
        transform = profile_new['transform']
        return patch_concat,profile_new


    def get_available_rows(self,mem_allocate,file_num,image_height,image_width,band_num=1,dtype=np.float32):
        # mem_allocate: GiB
        if mem_allocate > 512:
            Warning(f'Are you sure to allocate {mem_allocate}GiB memory?')
            print(f'Are you sure to allocate {mem_allocate}GiB memory?')
            pause()
        mem_allocate = self.GiByte_to_Byte(mem_allocate)

        memory_info = psutil.virtual_memory()
        total_mem  = self.sizeof_fmt(memory_info.total)
        sys_available_mem = memory_info.available
        if mem_allocate * 2 > memory_info.total:
            print('Memory not enough!!!','\ntotal mem:',total_mem,'\navailable mem:',self.sizeof_fmt(sys_available_mem))
            exit()
        if mem_allocate * 2 > memory_info.available:
            print('Memory Stress!!!','\ntotal mem:',total_mem,'\navailable mem:',self.sizeof_fmt(sys_available_mem))
            pause()
        array_init = np.zeros((1, 1), dtype=dtype)
        obj_mem = sys.getsizeof(array_init) - 128
        available_rows = int(mem_allocate / obj_mem / file_num / image_width)
        # print(mem_allocate / obj_mem / file_num / image_width)
        if available_rows < 1:
            raise Exception('memory not enough, please allocate more memory')
        if available_rows > image_height:
            print(f'Do not need that much memory, available_rows:{available_rows}, image_height:{image_height}')
            available_rows = image_height
        return available_rows

    def sizeof_fmt(self, num, suffix="B"):
        for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"

    def GiByte_to_Byte(self, GiByte):
        try:
            GiByte = float(GiByte)
        except:
            raise Exception('input error')
        return int(GiByte * 1024 * 1024 * 1024)

    def get_image_profiles(self,fpath):
        with rasterio.open(fpath) as src:
            profile = src.profile
            return profile

    def transform_to_block(self,outdir,njob=8):
        T.mkdir(outdir)
        flist = self.flist
        band_name_list = []
        for fpath in flist:
            fpath_obj = Path(fpath)
            band_name = fpath_obj.name
            band_name_list.append(band_name)
        if njob == 1:
            for idx in tqdm(self.block_index_list,desc='transform to block'):
                patch_concat,profile_new = self.array_iterator_index(idx)
                outf = join(outdir,f'{self.get_digit_str(self.iter_length,idx)}.tif')
                RasterIO_Func().write_tif_multi_bands(patch_concat, outf, profile_new, band_name_list)
        else:
            params_list = []
            for idx in self.block_index_list:
                params = [outdir,idx,band_name_list]
                params_list.append(params)
            MULTIPROCESS(self.kernel_transform_to_block,params_list).run(process=njob)
        pass

    def kernel_transform_to_block(self,params):
        outdir,idx,band_name_list = params
        patch_concat, profile_new = self.array_iterator_index(idx)
        outf = join(outdir, f'{self.get_digit_str(self.iter_length, idx)}.tif')
        RasterIO_Func().write_tif_multi_bands(patch_concat, outf, profile_new, band_name_list)

    def transform_to_spatial_dict(self,outdir,njob=8):
        T.mkdir(outdir)
        flist = self.flist
        band_name_list = []
        for fpath in flist:
            fpath_obj = Path(fpath)
            band_name = fpath_obj.name
            band_name_list.append(band_name)
        if njob == 1:
            for idx in tqdm(self.block_index_list,desc='transform to spatial dict'):
                patch_concat,profile_new = self.array_iterator_index(idx)
                row_size = patch_concat.shape[1]
                col_size = patch_concat.shape[2]
                spatial_dict = {}
                for r in range(row_size):
                    for c in range(col_size):
                        vals = patch_concat[:,r,c]
                        spatial_dict[(r+idx*self.available_rows,c)] = vals
                outf = join(outdir,f'{self.get_digit_str(self.iter_length,idx)}.npy')
                T.save_npy(spatial_dict,outf)
        else:
            params_list = []
            for idx in self.block_index_list:
                params = [outdir,idx]
                params_list.append(params)
            MULTIPROCESS(self.kernel_transform_to_spatial_dict,params_list).run(process=njob)

    def kernel_transform_to_spatial_dict(self,params):
        outdir, idx = params
        patch_concat, profile_new = self.array_iterator_index(idx)
        row_size = patch_concat.shape[1]
        col_size = patch_concat.shape[2]
        spatial_dict = {}
        for r in range(row_size):
            for c in range(col_size):
                vals = patch_concat[:, r, c]
                spatial_dict[(r + idx * self.available_rows, c)] = vals
        outf = join(outdir, f'{self.get_digit_str(self.iter_length, idx)}.npy')
        T.save_npy(spatial_dict, outf)

    def get_digit_str(self,total_len,idx):
        digit = math.log(total_len, 10) + 1
        digit = int(digit)
        digit_str = f'{idx:0{digit}d}'
        return digit_str

    def check_tifs(self):
        failed_flist = []
        for fpath in self.flist:
            try:
                with rasterio.open(fpath) as src:
                    profile = src.profile
                    # print(profile)
            except Exception as e:
                print(f'check tif error:{fpath}')
                print(e)
                print('----')
                failed_flist.append(fpath)
        return failed_flist
        pass


class RasterIO_Func:

    def __init__(self):

        pass

    def write_tif(self, array, outf, profile):
        with rasterio.open(outf, "w", **profile) as dst:
            dst.write(array, 1)

    def write_tif_multi_bands(self, array_3d, outf, profile, bands_description: list = None):
        dimension = array_3d.ndim
        if dimension == 2:
            array_3d = array_3d[np.newaxis, ... ]
        profile.update(count=array_3d.shape[0])
        with rasterio.open(outf, "w", **profile) as dst:
            for i in range(array_3d.shape[0]):
                dst.write(array_3d[i], i + 1)
                if bands_description is not None:
                    dst.set_band_description(i + 1, bands_description[i])

    def read_tif(self,fpath):
        with rasterio.open(fpath) as src:
            data = src.read()
            profile = src.profile
            data = data.squeeze()
            return data,profile

    def crop_tif(self,fpath,outf,in_shp):
        with rasterio.open(fpath) as src:
            shapes = gpd.read_file(in_shp).geometry
            subset, subset_transform = mask(src, shapes, crop=True)

            profile = src.profile
            profile.update({
                "height": subset.shape[1],
                "width": subset.shape[2],
                "transform": subset_transform
            })

        with rasterio.open(outf, "w", **profile) as dst:
            dst.write(subset)

    def mosaic_arrays(self,array_list,profile_list):
        datasets = []
        for arr, prof in zip(array_list, profile_list):
            if arr.ndim == 2 and prof["count"] == 1:
                arr = arr[np.newaxis, :, :]

            if arr.ndim == 2:
                prof.update(count=1)
            elif arr.ndim == 3:
                prof.update(count=arr.shape[0])
            else:
                raise ValueError("Invalid array dimensions for rasterio write")
            memfile = MemoryFile()
            with memfile.open(**prof) as dataset:
                dataset.write(arr)
            datasets.append(memfile.open())
        mosaic, mosaic_transform = merge(datasets)
        out_profile = profile_list[0].copy()
        out_profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform
        })
        return mosaic,out_profile

    def mosaic_tifs(self,flist,outf):
        array_list = []
        profile_list = []
        for fpath in flist:
            array,profile = self.read_tif(fpath)
            array_list.append(array)
            profile_list.append(profile)
        mosaic,out_profile = self.mosaic_arrays(array_list,profile_list)
        self.write_tif_multi_bands(mosaic, outf, out_profile)

        pass

    def get_tif_bounds(self,fpath):
        array,profile = self.read_tif(fpath)
        crs = profile['crs']
        originX = profile['transform'][2]
        originY = profile['transform'][5]
        pixelWidth = profile['transform'][0]
        pixelHeight = profile['transform'][4]
        endX = originX + array.shape[1] * pixelWidth
        endY = originY + array.shape[0] * pixelHeight
        ll_point = (originX, originY)
        lr_point = (endX, originY)
        ur_point = (endX, endY)
        ul_point = (originX, endY)
        return ll_point,lr_point,ur_point,ul_point

    def reproject_tif(self,fpath,outf,dst_crs,dst_crs_res=None):
        with rasterio.open(fpath) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,resolution=dst_crs_res
            )
            profile = src.profile
            profile.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
            })
            with rasterio.open(outf, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )



def main():
    # Download().run()
    Preprocess_HLS().run()
    # Download_From_GEE_1km().run()
    pass

if __name__ == '__main__':

    main()