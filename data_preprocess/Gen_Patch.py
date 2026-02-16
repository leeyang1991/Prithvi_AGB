from __init__ import *
from __global__ import *
import rasterio

this_script_root = join(data_root,'Patch')

class GenPatch:
    def __init__(self):
        self.data_dir = join(this_script_root,'Gen_patch')
        self.region = 'AZ'
        # self.region = 'NM'
        self.resolution = '30m'
        # self.resolution = '1km'
        self.PATCH_SIZE_1km = 224
        self.STRIDE_1km = 28

        # self.PATCH_SIZE_30m = 896
        # self.STRIDE_30m = 448

        self.PATCH_SIZE_30m = 1792
        self.STRIDE_30m = 896
        pass

    def run(self):
        self.generate_patches_HLS()
        # self.generate_patches_GEDI()
        # self.split_dataset() # deprecated, npy and json are not used
        # self.test_patch()
        pass

    @Decorator.shutup_gdal
    def generate_patches_HLS(self):
        import concat_data
        dstSRS = global_gedi_WKT()
        hls_path = join(concat_data.Concat_Data().data_dir, self.resolution,f'{self.region}_concat_{self.resolution}.tif')
        # outdir_hls = join(self.data_dir,'patches/hls_1km')
        outdir_hls = join(self.data_dir,'HLS',self.region,self.resolution)
        T.mkdir(outdir_hls,force=True)
        for f in tqdm(T.listdir(outdir_hls),desc='removing old patch'):
            fpath = join(outdir_hls,f)
            os.remove(fpath)
        # exit()

        if self.resolution == '1km':
            PATCH_SIZE = self.PATCH_SIZE_1km
            STRIDE = self.STRIDE_1km
        elif self.resolution == '30m':
            PATCH_SIZE = self.PATCH_SIZE_30m
            STRIDE = self.STRIDE_30m
        else:
            raise ValueError(f"Invalid resolution: {self.resolution}")
        random.seed(42)
        with rasterio.open(hls_path) as src_hls:
            h, w = src_hls.height, src_hls.width
            print(h,w)
            # exit()
            ds = gdal.Open(hls_path)
            gt = ds.GetGeoTransform()
            xres = gt[1]
            yres = gt[5]
            x_min = gt[0]
            y_max = gt[3]
            x_max = x_min + ds.RasterXSize * gt[1]
            y_min = y_max + ds.RasterYSize * gt[5]
            total_count = 0
            for row in range(0, h - PATCH_SIZE, STRIDE):
                for col in range(0, w - PATCH_SIZE, STRIDE):
                    total_count += 1
            print('total_count',total_count)
        params_list = []
        count = 0
        for row in tqdm(range(0, h - PATCH_SIZE, STRIDE), desc="Sliding rows"):
        # for row in tqdm(range(0, 1), desc="Sliding rows"):
            for col in range(0, w - PATCH_SIZE, STRIDE):
                digit_str = self.get_digit_str(total_count, count)
                patch_name = f"patch_{digit_str}"
                patch_fpath = join(outdir_hls, patch_name + '.tif')
                count += 1
                params = [hls_path,row,col,x_min,xres,y_max,row,yres,patch_fpath,PATCH_SIZE,dstSRS]
                params_list.append(params)
                # self.kernrl_generate_patches_HLS(params)
        MULTIPROCESS(self.kernrl_generate_patches_HLS,params_list).run(process=10,process_or_thread='p')
        print(f"Total patches saved: {count}")

    def kernrl_generate_patches_HLS(self,params):
        hls_path,row,col,x_min,xres,y_max,row,yres,patch_fpath,PATCH_SIZE,dstSRS = params
        with rasterio.open(hls_path) as src_hls:
            hls_patch = src_hls.read(
                window=((row, row + PATCH_SIZE), (col, col + PATCH_SIZE))
            ).astype(np.float32)
            for band in hls_patch:
                band[band < -9999] = np.nan
            # x_min_all, xres, 0, y_max_all, 0, yres
            x_min_i = x_min + col * xres
            y_max_i = y_max + row * yres
            self.patch_to_tif(patch_fpath, hls_patch, PATCH_SIZE, x_min_i, xres, y_max_i, yres, dstSRS)

        pass


    def patch_to_tif(self,patch_fpath, hls_patch,PATCH_SIZE,x_min, xres, y_max, yres,dstSRS):
        # pprint(hls_patch)
        # print(np.shape(hls_patch))
        # exit()
        # bands_name_list = global_band_list
        bands_name_list = global_band_list_8
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(patch_fpath, PATCH_SIZE, PATCH_SIZE, len(hls_patch), gdal.GDT_Float32,
                               options=['COMPRESS=LZW', 'BIGTIFF=YES'])
        # out_ds = driver.Create(outf,
        #                        src0.RasterXSize,
        #                        src0.RasterYSize,
        #                        len(tif_list),
        #                        gdal.GDT_Float32)
        out_gt = (x_min, xres, 0, y_max, 0, yres)
        out_ds.SetGeoTransform(out_gt)
        out_ds.SetProjection(dstSRS)
        for idx, band in enumerate(hls_patch, start=1):
            # band = hls_patch[idx-1]
            out_ds.GetRasterBand(idx).WriteArray(band)
            out_ds.GetRasterBand(idx).SetDescription(bands_name_list[idx - 1])
            out_ds.GetRasterBand(idx).SetNoDataValue(-999999)
        # out_band.FlushCache()

        out_ds = None
        # exit()
        pass

    @Decorator.shutup_gdal
    def generate_patches_GEDI(self):
        import GEDI
        gedi_path = join(GEDI.Preprocess_GEDI().data_dir, 'crop', f'{self.region}.tif')
        outdir_gedi = join(self.data_dir,'GEDI',self.region,self.resolution)
        T.mkdir(outdir_gedi,force=True)
        dstSRS = global_gedi_WKT()
        PATCH_SIZE = self.PATCH_SIZE_1km
        STRIDE = self.STRIDE_1km
        random.seed(42)
        with rasterio.open(gedi_path) as src_hls:
            h, w = src_hls.height, src_hls.width
            print(h, w)
            # exit()
            ds = gdal.Open(gedi_path)
            gt = ds.GetGeoTransform()
            xres = gt[1]
            yres = gt[5]
            x_min = gt[0]
            y_max = gt[3]
            x_max = x_min + ds.RasterXSize * gt[1]
            y_min = y_max + ds.RasterYSize * gt[5]

            total_count = 0
            for row in range(0, h - PATCH_SIZE, STRIDE):
                # for row in tqdm(range(0, 1), desc="Sliding rows"):
                for col in range(0, w - PATCH_SIZE, STRIDE):
                    total_count += 1
            print(f"Total patches saved: {total_count}")

            count = 0
            for row in tqdm(range(0, h - PATCH_SIZE, STRIDE), desc="Sliding rows"):
            # for row in tqdm(range(0, 1), desc="Sliding rows"):
                for col in range(0, w - PATCH_SIZE, STRIDE):
                    gedi_patch = src_hls.read(
                        window=((row, row + PATCH_SIZE), (col, col + PATCH_SIZE))
                    ).astype(np.float32)
                    # 检查是否全NaN
                    # if np.isnan(gedi_patch).all():
                    #     continue
                    for band in gedi_patch:
                        band[band<-9999] = np.nan

                    patch_name = self.get_digit_str(total_count,count)
                    patch_fpath = join(outdir_gedi,f'patch_{patch_name}.tif')
                    # x_min_all, xres, 0, y_max_all, 0, yres
                    x_min_i = x_min + col * xres
                    y_max_i = y_max + row * yres
                    self.patch_to_tif(patch_fpath, gedi_patch, PATCH_SIZE, x_min_i, xres, y_max_i, yres,dstSRS)
                    count += 1

        print(f"Total patches saved: {count}")


    def split_dataset(self,train_ratio=0.8):
        fdir_json = join(self.data_dir,'patches/json')
        fdir_npy = join(self.data_dir,'patches/npy')
        outdir = join(self.data_dir,'csv')
        T.mkdir(outdir,force=True)
        samples = []
        for f in T.listdir(fdir_json):
            if not f.endswith('.json'):
                continue
            patch_name = f.split('.')[0]
            json_fpath = join(fdir_json,f)
            agb_mean = json.load(open(json_fpath,'r'))['AGB_mean']
            samples.append((patch_name, agb_mean))
        random.shuffle(samples)
        n_train = int(len(samples) * train_ratio)
        train, val = samples[:n_train], samples[n_train:]

        def save_list(samples, fname):
            with open(join(outdir,fname), "w") as f:
                for name, agb in samples:
                    f.write(f"{fdir_npy}/{(name + '.npy')},{agb}\n")

        save_list(train, "train_list.csv")
        save_list(val, "validate_list.csv")

        print(f"Train: {len(train)}, Validate: {len(val)}")

    def test_patch(self):
        import numpy as np, json, matplotlib.pyplot as plt
        patch_fpath = join(self.data_dir,'patches/npy','patch_00010.npy')
        json_fpath = join(self.data_dir,'patches/json','patch_00010.json')
        patch = np.load(patch_fpath)
        label = json.load(open(json_fpath))['AGB_mean']
        plt.imshow(patch[2], cmap='jet')  # 可视化红波段
        plt.colorbar()
        plt.title(f"AGB={label:.1f}")
        plt.show()
        pass

    def get_digit_str(self,total_len,idx):
        digit = math.log(total_len, 10) + 1
        digit = int(digit)
        digit_str = f'{idx:0{digit}d}'
        return digit_str


class GenPatch_annual:
    def __init__(self):
        self.data_dir = join(this_script_root,'GenPatch_annual')
        self.region = 'AZ'
        # self.region = 'NM'
        self.resolution = '30m'
        # self.resolution = '1km'
        self.PATCH_SIZE_1km = 224
        self.STRIDE_1km = 28

        # self.PATCH_SIZE_30m = 896
        # self.STRIDE_30m = 448

        self.PATCH_SIZE_30m = 1792
        self.STRIDE_30m = 896
        pass

    def run(self):
        self.generate_patches_HLS()
        pass

    @Decorator.shutup_gdal
    def generate_patches_HLS(self):
        import concat_data
        dstSRS = global_gedi_WKT()
        for year in ['2019','2020','2021','2022','2023']:
            print(year)
            hls_path = join(concat_data.Concat_Data_annual().data_dir, self.resolution,f'{year}/{self.region}_concat_{self.resolution}.tif')
            # outdir_hls = join(self.data_dir,'patches/hls_1km')
            outdir_hls = join(self.data_dir,'HLS',self.region,self.resolution,year)
            T.mkdir(outdir_hls,force=True)
            for f in tqdm(T.listdir(outdir_hls),desc='removing old patch'):
                fpath = join(outdir_hls,f)
                os.remove(fpath)
            # exit()

            if self.resolution == '1km':
                PATCH_SIZE = self.PATCH_SIZE_1km
                STRIDE = self.STRIDE_1km
            elif self.resolution == '30m':
                PATCH_SIZE = self.PATCH_SIZE_30m
                STRIDE = self.STRIDE_30m
            else:
                raise ValueError(f"Invalid resolution: {self.resolution}")
            random.seed(42)
            # print(hls_path)
            # print(isfile(hls_path))
            # exit()
            with rasterio.open(hls_path) as src_hls:
                h, w = src_hls.height, src_hls.width
                print(h,w)
                # exit()
                ds = gdal.Open(hls_path)
                gt = ds.GetGeoTransform()
                xres = gt[1]
                yres = gt[5]
                x_min = gt[0]
                y_max = gt[3]
                x_max = x_min + ds.RasterXSize * gt[1]
                y_min = y_max + ds.RasterYSize * gt[5]
                total_count = 0
                for row in range(0, h - PATCH_SIZE, STRIDE):
                    for col in range(0, w - PATCH_SIZE, STRIDE):
                        total_count += 1
                print('total_count',total_count)
            params_list = []
            count = 0
            for row in tqdm(range(0, h - PATCH_SIZE, STRIDE), desc="Sliding rows"):
            # for row in tqdm(range(0, 1), desc="Sliding rows"):
                for col in range(0, w - PATCH_SIZE, STRIDE):
                    digit_str = self.get_digit_str(total_count, count)
                    patch_name = f"patch_{digit_str}"
                    patch_fpath = join(outdir_hls, patch_name + '.tif')
                    count += 1
                    params = [hls_path,row,col,x_min,xres,y_max,row,yres,patch_fpath,PATCH_SIZE,dstSRS]
                    params_list.append(params)
                    # self.kernrl_generate_patches_HLS(params)
            MULTIPROCESS(self.kernrl_generate_patches_HLS,params_list).run(process=30,process_or_thread='p')
            print(f"Total patches saved: {count}")

    def kernrl_generate_patches_HLS(self,params):
        hls_path,row,col,x_min,xres,y_max,row,yres,patch_fpath,PATCH_SIZE,dstSRS = params
        with rasterio.open(hls_path) as src_hls:
            hls_patch = src_hls.read(
                window=((row, row + PATCH_SIZE), (col, col + PATCH_SIZE))
            ).astype(np.float32)
            for band in hls_patch:
                band[band < -9999] = np.nan
            # x_min_all, xres, 0, y_max_all, 0, yres
            x_min_i = x_min + col * xres
            y_max_i = y_max + row * yres
            self.patch_to_tif(patch_fpath, hls_patch, PATCH_SIZE, x_min_i, xres, y_max_i, yres, dstSRS)

        pass


    def patch_to_tif(self,patch_fpath, hls_patch,PATCH_SIZE,x_min, xres, y_max, yres,dstSRS):
        # pprint(hls_patch)
        # print(np.shape(hls_patch))
        # exit()
        # bands_name_list = global_band_list
        bands_name_list = global_band_list_8
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(patch_fpath, PATCH_SIZE, PATCH_SIZE, len(hls_patch), gdal.GDT_Float32,
                               options=['COMPRESS=LZW', 'BIGTIFF=YES'])
        # out_ds = driver.Create(outf,
        #                        src0.RasterXSize,
        #                        src0.RasterYSize,
        #                        len(tif_list),
        #                        gdal.GDT_Float32)
        out_gt = (x_min, xres, 0, y_max, 0, yres)
        out_ds.SetGeoTransform(out_gt)
        out_ds.SetProjection(dstSRS)
        for idx, band in enumerate(hls_patch, start=1):
            # band = hls_patch[idx-1]
            out_ds.GetRasterBand(idx).WriteArray(band)
            out_ds.GetRasterBand(idx).SetDescription(bands_name_list[idx - 1])
            out_ds.GetRasterBand(idx).SetNoDataValue(-999999)
        # out_band.FlushCache()

        out_ds = None
        # exit()
        pass

    def get_digit_str(self,total_len,idx):
        digit = math.log(total_len, 10) + 1
        digit = int(digit)
        digit_str = f'{idx:0{digit}d}'
        return digit_str

class Split_Patch:

    def __init__(self):
        self.data_dir = join(this_script_root,'Split_patch')
        self.region = 'AZ'
        self.resolution = '1km'
        pass

    def run(self):
        self.HLS()
        self.GEDI()
        pass

    def gen_random_path_list(self):
        random.seed(42)
        outdir = join(self.data_dir,'patches')
        fpath_dir = join(GenPatch().data_dir,f'GEDI/{self.region}/{self.resolution}')
        fpath_list = T.listdir(fpath_dir)
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        random.shuffle(fpath_list)
        train_flist = fpath_list[:int(len(fpath_list)*train_ratio)]
        val_flist = fpath_list[int(len(fpath_list)*train_ratio):int(len(fpath_list)*(train_ratio+val_ratio))]
        test_flist = fpath_list[int(len(fpath_list)*(train_ratio+val_ratio)):]

        return train_flist, val_flist, test_flist

    def HLS(self):
        train_flist, val_flist, test_flist = self.gen_random_path_list()
        patch_fdir = join(GenPatch().data_dir,'HLS',self.region,self.resolution)
        train_fdir = join(self.data_dir,'HLS',self.region,self.resolution,'train')
        val_fdir = join(self.data_dir,'HLS',self.region,self.resolution,'val')
        test_fdir = join(self.data_dir,'HLS',self.region,self.resolution,'test')
        T.mkdir(train_fdir,force=True)
        T.mkdir(val_fdir,force=True)
        T.mkdir(test_fdir,force=True)

        for f in tqdm(train_flist,desc='train'):
            src_fpath = join(patch_fdir,f)
            dst_fpath = join(train_fdir,f)
            shutil.copy(src_fpath,dst_fpath)

        for f in tqdm(val_flist,desc='val'):
            src_fpath = join(patch_fdir,f)
            dst_fpath = join(val_fdir,f)
            shutil.copy(src_fpath,dst_fpath)

        for f in tqdm(test_flist,desc='test'):
            src_fpath = join(patch_fdir,f)
            dst_fpath = join(test_fdir,f)
            shutil.copy(src_fpath,dst_fpath)
        pass

    def GEDI(self):
        train_flist, val_flist, test_flist = self.gen_random_path_list()

        patch_fdir = join(GenPatch().data_dir, 'GEDI', self.region, self.resolution)
        train_fdir = join(self.data_dir, 'GEDI', self.region, self.resolution, 'train')
        val_fdir = join(self.data_dir, 'GEDI', self.region, self.resolution, 'val')
        test_fdir = join(self.data_dir, 'GEDI', self.region, self.resolution, 'test')

        T.mkdir(train_fdir,force=True)
        T.mkdir(val_fdir,force=True)
        T.mkdir(test_fdir,force=True)

        for f in tqdm(train_flist,desc='train'):
            src_fpath = join(patch_fdir,f)
            dst_fpath = join(train_fdir,f)
            shutil.copy(src_fpath,dst_fpath)

        for f in tqdm(val_flist,desc='val'):
            src_fpath = join(patch_fdir,f)
            dst_fpath = join(val_fdir,f)
            shutil.copy(src_fpath,dst_fpath)

        for f in tqdm(test_flist,desc='test'):
            src_fpath = join(patch_fdir,f)
            dst_fpath = join(test_fdir,f)
            shutil.copy(src_fpath,dst_fpath)
        pass

def main():
    # GenPatch().run()
    GenPatch_annual().run()
    # Split_Patch().run()
    pass

if __name__ == '__main__':

    main()