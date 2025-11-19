import copy

import affine
import math

import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import load_entry_point
from torch.utils.benchmark import ordered_unique
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
import geopandas as gpd
from __init__ import *

from __global__ import *
import rasterio
import geopandas as gp
import earthaccess
import ee
import urllib3
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

        pass

    def run(self):
        # self.gen_urls_test()
        # self.gen_urls()
        # self.download()
        # self.check_download()
        # self.move_tile_to_different_folder()
        self.delete_empty_folders()
        pass

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
        outdir = join(self.data_dir, 'url_list')
        T.mkdir(outdir, force=True)
        outf = join(outdir, 'santarita.txt')
        if isfile(outf):
            return
        earthaccess.login(persist=True)
        # earthaccess.download()
        field = gp.read_file('santarita.geojson')
        bbox = tuple(list(field.total_bounds))
        # print(bbox)
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
                fw.write(url + '\n')
        fw.close()
        pass

    def download(self):
        # todo: split tiles in different folders
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
        with open(join(self.data_dir, 'url_list', 'santarita.txt'), 'r') as f:
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
            params_list.append([url,session,outdir])
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
            fw = open(outf,'wb')

            r = session.get(url,stream=True)
            for chunk in r.iter_content(chunk_size=8192):
                fw.write(chunk)
            fw.close()
        except Exception as e:
            print('error')
            print(e)
            print('--------')
            return

    @Decorator.shutup_gdal
    def check_download(self):
        fdir = join(self.data_dir,'Download')
        for folder in tqdm(T.listdir(fdir)):
            for f in T.listdir(join(fdir,folder)):
                fpath = join(fdir,folder,f)
                try:
                    raster = gdal.Open(fpath)
                    raster.GetProjection()
                except:
                    print(fpath)
                    os.remove(fpath)
        pass

    def move_tile_to_different_folder(self):
        fdir = join(self.data_dir,'Download')
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

        fdir = join(self.data_dir,'Download')
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
        # self.re_proj()
        self.quality_control_long_term_mean()
        # self.image_shp()
        # self.concatenate()
        # self.plot_time_series()
        # self.aggragate()
        # self.get_tif_template()
        # self.spatial_dict_to_tif()
        # self.mosaic_spatial_tifs()
        # self.merge_bands()
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
        import GEDI
        wkt_dest = GEDI.Preprocess_GEDI().get_WKT()
        fdir = join(self.data_dir,'Download')
        outdir = join(self.data_dir,'reproj')
        T.mkdir(outdir)
        params_list = []
        invalid_fpath = []
        for folder in tqdm(T.listdir(fdir)):
            outdir_i = join(outdir,folder)
            for f in T.listdir(join(fdir,folder)):
                fpath = join(fdir,folder,f)
                outpath = join(outdir_i,f)
                # print(outpath)
                try:
                    src_wkt = self.get_WKT(fpath)
                except:
                    print(fpath)
                    invalid_fpath.append(fpath)
                    src_wkt = None
                # ToRaster().resample_reproj(fpath, outpath, 30, srcSRS=src_wkt, dstSRS=wkt_dest)
                params_list.append([fpath, outpath, src_wkt, wkt_dest,outdir_i])
        # pprint(invalid_fpath)
        MULTIPROCESS(self.kernel_re_proj,params_list).run(process=30)

    def kernel_re_proj(self,params):
        fpath, outpath, src_wkt, wkt_dest,outdir_i = params
        try:
            T.mkdir(outdir_i, force=True)
        except:
            pass

        ToRaster().resample_reproj(fpath, outpath, 30, srcSRS=src_wkt, dstSRS=wkt_dest)

    def get_WKT(self,fpath):
        raster = gdal.Open(fpath)
        projection_wkt = raster.GetProjection()
        return projection_wkt

    def quality_control_long_term_mean(self):
        outdir = join(self.data_dir,'quality_control_long_term_mean')
        T.mkdir(outdir,force=True)
        njob = 32
        memory_allocate = 0.2 # in GB, 6nGB, njob = tot mem/6nGB
        qa_filter_list = self.get_qa_filter_list()
        band_list = self.bands_list
        fdir = join(self.data_dir,'Download')
        params_list = []
        for tile in T.listdir(fdir):
            dtype = np.int16
            # get QA 3d array
            flist_qa = []
            for folder in T.listdir(join(fdir,tile)):
                qa_fpath = join(fdir,tile, folder, f'{folder}.Fmask.tif')
                flist_qa.append(qa_fpath)
            Tif_loader_qa = Tif_loader(flist_qa,memory_allocate,dtype=dtype)

            # get each band 3d array
            for band in band_list:
                band_fpath_list = []
                for folder in T.listdir(join(fdir, tile)):
                    band_fpath = join(fdir, tile, folder, f'{folder}.{band}.tif')
                    band_fpath_list.append(band_fpath)
                Tif_loader_band = Tif_loader(band_fpath_list,memory_allocate,dtype=dtype)
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
        band_patch_concat[~qa_filter_mask_concat] = -9999
        band_patch_concat = np.array(band_patch_concat, dtype=np.float32)
        band_patch_concat[band_patch_concat < -999] = np.nan
        band_patch_concat_average = np.nanmean(band_patch_concat, axis=0)
        try:
            T.mkdir(outdir_i, force=True)
        except:
            pass
        RasterIO_Func().write_tif(band_patch_concat_average, outf, band_profile)


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

    @Decorator.shutup_gdal
    def image_shp(self):
        from shapely.geometry import Polygon
        import geopandas as gpd
        outdir = join(self.data_dir,'image_shp')
        T.mkdir(outdir)
        fdir = join(self.data_dir,'reproj_qa')
        pos_dict = {}
        for folder in tqdm(T.listdir(fdir)):
            pos_code = folder.split('.')[2]
            sat_code = folder.split('.')[1]
            if sat_code+pos_code in pos_dict:
                continue
            for f in T.listdir(join(fdir,folder)):
                fpath = join(fdir,folder,f)
                array, originX, originY, pixelWidth, pixelHeight, _ = self.raster2array(fpath)
                pos_dict[sat_code+pos_code] = {'originX':originX,'originY':originY,'endX':originX+array.shape[1]*pixelWidth,'endY':originY+array.shape[0]*pixelHeight}
                break

        rectangle_list = []
        name_list = []
        for pos_code in pos_dict:
            print(pos_code)
            originX = pos_dict[pos_code]['originX']
            originY = pos_dict[pos_code]['originY']
            endX = pos_dict[pos_code]['endX']
            endY = pos_dict[pos_code]['endY']
            ll_point = (originX, originY)
            lr_point = (endX, originY)
            ur_point = (endX, endY)
            ul_point = (originX, endY)
            polygon_geom = Polygon([ll_point, lr_point, ur_point, ul_point])
            rectangle_list.append(polygon_geom)
            name_list.append(pos_code)

        outf = join(outdir, 'sites.shp')
        crs = {'init': 'epsg:6933'}  # 设置坐标系
        polygon = gpd.GeoDataFrame(crs=crs, geometry=rectangle_list)  # 将多边形对象转换为GeoDataFrame对象
        polygon['name'] = name_list
        polygon.to_file(outf)

    def concatenate(self):
        # eat more than 100GB memory
        fdir = join(self.data_dir, 'reproj_qa')
        outdir = join(self.data_dir,'reproj_qa_concatenate')
        T.mkdir(outdir)

        selected_bands = [
            'B02',
            'B03',
            'B04',
            'B05',
            'B06',
            'B07',
        ]
        year_list = [2019,2020,2021,2022,2023]
        sat_list = ['L30','S30']

        params_list = []
        for sat in sat_list:
            outdir_sat = join(outdir,sat)
            for band in selected_bands:
                outdir_band = join(outdir_sat,band)
                for year in year_list:
                    params = fdir, outdir_band, sat, band, year
                    params_list.append(params)
        MULTIPROCESS(self.kernel_concatenate,params_list).run(8)

    def kernel_concatenate(self,params):
        fdir, outdir_band, sat, band, year = params
        outdir_year = join(outdir_band, str(year))
        pos_dict = {}
        for folder in tqdm(T.listdir(fdir)):
            pos_code = folder.split('.')[2]
            if not pos_code in pos_dict:
                pos_dict[pos_code] = []
            for f in T.listdir(join(fdir, folder)):
                sat_ = f.split('.')[1]
                band_ = f.split('.')[-2]
                year_ = f.split('.')[3].split('T')[0][:4]
                year_ = int(year_)
                if not sat == sat_:
                    continue
                if not band == band_:
                    continue
                if not year == year_:
                    continue
                fpath = join(fdir, folder, f)
                pos_dict[pos_code].append(join(fdir, folder, f))
        for pos_code in pos_dict:
            outdir_pos = join(outdir_year, pos_code)
            try:
                T.mkdir(outdir_pos, force=True)
            except:
                pass
            fpath_list = pos_dict[pos_code]
            self.data_transform(fpath_list, outdir_pos)

        pass

    def data_transform(self, fpath_list,outdir, n=100000):
        n = int(n)
        Tools().mkdir(outdir)
        # per_pix_data
        # fpath_list
        all_array = []
        # for d in date_list:
        for fpath in tqdm(fpath_list,desc='loading tifs'):
            if fpath.endswith('.tif'):
                # print(d)
                array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                array = np.array(array, dtype=float)
                all_array.append(array)

        row = len(all_array[0])
        col = len(all_array[0][0])
        all_array = np.array(all_array)
        all_array = all_array.T
        void_dic = {}
        void_dic_list = []
        for r in tqdm(list(range(row)),desc='transforming'):
            for c in range(col):
                time_series = all_array[c][r]
                time_series = np.array(time_series)
                void_dic_list.append((r, c))
                void_dic[(r, c)] = time_series
        flag = 0
        temp_dic = {}
        for key in tqdm(void_dic_list, 'saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            arr = void_dic[key]
            arr = np.array(arr)
            temp_dic[key] = arr
            if flag % n == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + '/per_pix_dic_%05d' % (flag / n), temp_dic)
                temp_dic = {}
        np.save(outdir + '/per_pix_dic_%05d' % 0, temp_dic)

    def aggragate(self):
        fdir = join(self.data_dir,'reproj_qa_concatenate')
        outdir = join(self.data_dir,'reproj_qa_concatenate_aggragate')

        params_list = []
        for tile in self.tile_list:
            block_list = self.block_dict[tile]
            for block in block_list:
                params = [fdir,tile,block,outdir]
                params_list.append(params)

        MULTIPROCESS(self.kernel_agg,params_list).run(process=30)

    def kernel_agg(self,params):
        fdir,tile,block,outdir = params
        for band in self.bands_list:
            fpath_list = []
            for sat in self.sat_list:
                for year in self.year_list:
                    fpath = join(fdir, '/'.join([sat, band, year, tile, block]))
                    fpath_list.append(fpath)
            outdir_i = join(outdir, band, tile)
            try:
                T.mkdir(outdir_i, force=True)
            except:
                pass
            spatial_dict_all = {}
            for fpath in fpath_list:
                spatial_dict_i = T.load_npy(fpath)
                spatial_dict_all[fpath] = spatial_dict_i
            pix_list = spatial_dict_all[fpath_list[0]].keys()
            spatial_dict_mean = {}
            # for pix in tqdm(pix_list,desc=f'{band}_{tile}_{block}'):
            for pix in pix_list:
                vals_list = np.array([])
                for fpath in fpath_list:
                    vals_i = spatial_dict_all[fpath][pix]
                    vals_list = np.append(vals_list, vals_i)
                if T.is_all_nan(vals_list):
                    continue
                vals_mean = np.nanmean(vals_list)
                spatial_dict_mean[pix] = vals_mean
            outf = join(outdir_i, block)
            T.save_npy(spatial_dict_mean, outf)

        pass

    @Decorator.shutup_gdal
    def spatial_dict_to_tif(self):
        import GEDI
        GEDI_wkt = GEDI.Preprocess_GEDI().get_WKT()
        outdir = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif')
        T.mkdir(outdir)
        tile_list = self.tile_list
        band_list = self.bands_list
        tile_template_dict = self.get_tif_template()
        for band in band_list:
            outdir_i = join(outdir,band)
            T.mkdir(outdir_i)
            for tile in tile_list:
                print(band,tile,'\n')
                spatial_dict_dir = join(self.data_dir,'reproj_qa_concatenate_aggragate',band,tile)
                template_tif = tile_template_dict[tile]
                outf = join(outdir_i,tile+'.tif')
                D = DIC_and_TIF(tif_template=template_tif)

                spatial_dict = T.load_npy_dir(spatial_dict_dir)
                arr = D.pix_dic_to_spatial_arr(spatial_dict)
                _, originX, originY, pixelWidth, pixelHeight, _ = self.raster2array(template_tif)
                self.array2raster(outf, originX, originY, pixelWidth, pixelHeight, arr, GEDI_wkt)
        pass

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

    @Decorator.shutup_gdal
    def mosaic_spatial_tifs(self):

        import GEDI
        projection_wkt = GEDI.Preprocess_GEDI().get_WKT()
        fdir = join(self.data_dir, 'reproj_qa_concatenate_aggragate_tif')
        outdir = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic')
        T.mkdir(outdir)
        tile_list = self.tile_list
        bands_list = self.bands_list
        for band in bands_list:
            print(band)
            fpath_list = []
            for tile in tile_list:
                fpath = join(fdir,band,tile+'.tif')
                fpath_list.append(fpath)
            outf = join(outdir,f'{band}.tif')
            self.__mosaic_spatial_tifs(fpath_list,outf,projection_wkt)
        pass

    def __mosaic_spatial_tifs(self,tif_list, output_path, dstSRS, output_format="GTiff", nodata_value=-999999):
        # todo: add to lytools
        ref_ds = gdal.Open(tif_list[0])
        gt = ref_ds.GetGeoTransform()
        xres = gt[1]
        yres = gt[5]
        tiles_info = []

        for path in tif_list:
            ds = gdal.Open(path)
            gt_i = ds.GetGeoTransform()
            x_min = gt_i[0]
            y_max = gt_i[3]
            x_max = x_min + ds.RasterXSize * gt_i[1]
            y_min = y_max + ds.RasterYSize * gt_i[5]
            tiles_info.append((path, x_min, x_max, y_min, y_max))

        x_mins = [i[1] for i in tiles_info]
        x_maxs = [i[2] for i in tiles_info]
        y_mins = [i[3] for i in tiles_info]
        y_maxs = [i[4] for i in tiles_info]

        x_min_all, x_max_all = min(x_mins), max(x_maxs)
        y_min_all, y_max_all = min(y_mins), max(y_maxs)

        cols = int((x_max_all - x_min_all) / xres)
        rows = int((y_max_all - y_min_all) / abs(yres))

        mosaic = np.ones((rows, cols), dtype=np.float32) * nodata_value
        count = np.zeros((rows, cols), dtype=np.uint16)  # 记录每个像素被填充次数
        # nodata = nodata_value

        for path, xmin, xmax, ymin, ymax in tiles_info:
            ds = gdal.Open(path)
            data = ds.ReadAsArray().astype(np.float32)
            gt_i = ds.GetGeoTransform()

            # tile 在 mosaic 中的起始、结束行列号
            x_off = int((xmin - x_min_all) / xres)
            y_off = int((y_max_all - ymax) / abs(yres))
            h, w = ds.RasterYSize, ds.RasterXSize
            data[np.isnan(data)] = nodata_value

            mask = data != nodata_value  # 有效像素
            sub = mosaic[y_off:y_off+h, x_off:x_off+w]

            empty = sub == nodata_value
            fill_mask = mask & empty
            sub[fill_mask] = data[fill_mask]
            mosaic[y_off:y_off + h, x_off:x_off + w] = sub

        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32,
                               options=['COMPRESS=LZW', 'BIGTIFF=YES'])
        out_gt = (x_min_all, xres, 0, y_max_all, 0, yres)
        out_ds.SetGeoTransform(out_gt)
        out_ds.SetProjection(dstSRS)

        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(mosaic)
        out_band.SetNoDataValue(nodata_value)
        out_band.FlushCache()

        out_ds = None
        pass


    @Decorator.shutup_gdal
    def merge_bands(self,nodata_value=-999999):
        import GEDI
        projection_wkt = GEDI.Preprocess_GEDI().get_WKT()
        fdir = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic')
        outdir = join(self.data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands')
        T.mkdir(outdir)
        tile_list = self.tile_list
        tif_list = []
        bands_name_list = []
        for band in self.bands_list:
            fpath = join(fdir,band+'.tif')
            tif_list.append(fpath)
            bands_name_list.append(band)
        # print(tif_list)
        # print(bands_name_list)
        # exit()
        outf = join(outdir,'B2-B7.tif')

        src0 = gdal.Open(tif_list[0])
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(outf,
                               src0.RasterXSize,
                               src0.RasterYSize,
                               len(tif_list),
                               gdal.GDT_Float32)

        out_ds.SetGeoTransform(src0.GetGeoTransform())
        out_ds.SetProjection(projection_wkt)

        for idx, tif in enumerate(tif_list, start=1):
            src = gdal.Open(tif)
            band = src.GetRasterBand(1).ReadAsArray()
            out_ds.GetRasterBand(idx).WriteArray(band)
            out_ds.GetRasterBand(idx).SetDescription(bands_name_list[idx - 1])
            out_ds.GetRasterBand(idx).SetNoDataValue(nodata_value)

        out_ds.FlushCache()
        out_ds = None

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

    def __init__(self,flist,memory_allocate,dtype=None):
        self.flist = flist
        self.memory_allocate = memory_allocate
        self.profile = self.get_image_profiles(flist[0])
        # pprint(profile)
        self.h = self.profile['height']
        self.w = self.profile['width']
        if dtype == None:
            self.dtype = self.profile['dtype']
        else:
            self.dtype = dtype
        self.available_rows = self.get_available_rows(self.memory_allocate, len(flist), self.h, self.w, dtype=self.dtype)
        self.iter_length = math.ceil(self.h / self.available_rows)
        self.block_index_list = list(range(math.ceil(self.h / self.available_rows)))
        print('sliding rows',self.available_rows)
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

class RasterIO_Func:

    def __init__(self):

        pass

    def write_tif(self, array, outf, profile):
        with rasterio.open(outf, "w", **profile) as dst:
            dst.write(array, 1)

    def write_tif_multi_bands(self, array_3d, outf, profile, bands_description: list = None):
        with rasterio.open(outf, "w", **profile) as dst:
            for i in range(array_3d.shape[0]):
                dst.write(array_3d[i], i + 1)
                if bands_description is not None:
                    dst.set_band_description(i + 1, bands_description[i])

    def read_tif(self,fpath):
        with rasterio.open(fpath) as src:
            data = src.read()
            profile = src.profile
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

def main():
    # Download().run()
    Preprocess_HLS().run()
    # Download_From_GEE_1km().run()
    pass

if __name__ == '__main__':

    main()