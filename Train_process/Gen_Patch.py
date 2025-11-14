import shutil

import matplotlib.pyplot as plt

from __init__ import *
from __global__ import *
import rasterio

this_script_root = join(data_root,'Patch')

class GenPatch:
    def __init__(self):
        self.data_dir = this_script_root
        pass

    def run(self):
        self.generate_patches_HLS()
        # self.generate_patches_GEDI()
        # self.split_dataset() # deprecated, npy and json are not used
        # self.test_patch()
        pass

    @Decorator.shutup_gdal
    def generate_patches_HLS(self):
        import HLS, GEDI
        dstSRS = GEDI.Preprocess_GEDI().get_WKT()
        # hls_path = join(HLS.Preprocess_HLS().data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7_1km_224.tif')
        # hls_path = join(HLS.Preprocess_HLS().data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7_1km.tif')
        hls_path = join(HLS.Preprocess_HLS().data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7.tif')
        # outdir_hls = join(self.data_dir,'patches/hls_1km')
        outdir_hls = join(self.data_dir,'patches/hls_30m')
        T.mkdir(outdir_hls,force=True)

        # PATCH_SIZE = 64
        PATCH_SIZE = 224
        STRIDE = 112  # 可改成112表示50%重叠
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
            count = 0

            for row in tqdm(range(0, h - PATCH_SIZE, STRIDE), desc="Sliding rows"):
            # for row in tqdm(range(0, 1), desc="Sliding rows"):
                for col in range(0, w - PATCH_SIZE, STRIDE):
                    # --- 读取HLS patch ---
                    hls_patch = src_hls.read(
                        window=((row, row + PATCH_SIZE), (col, col + PATCH_SIZE))
                    ).astype(np.float32)
                    # 检查是否全NaN
                    if np.isnan(hls_patch).all():
                        continue
                    for band in hls_patch:
                        band[band<-9999] = np.nan

                    patch_name = f"patch_{count:05d}"
                    patch_fpath = join(outdir_hls,patch_name+'.tif')
                    # x_min_all, xres, 0, y_max_all, 0, yres
                    x_min_i = x_min + col * xres
                    y_max_i = y_max + row * yres
                    self.patch_to_tif(patch_fpath, hls_patch, PATCH_SIZE, x_min_i, xres, y_max_i, yres,dstSRS)
                    count += 1

        print(f"Total patches saved: {count}")

    def patch_to_tif(self,patch_fpath, hls_patch,PATCH_SIZE,x_min, xres, y_max, yres,dstSRS):
        # pprint(hls_patch)
        # print(np.shape(hls_patch))
        # exit()
        bands_name_list = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']
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
        import HLS, GEDI
        dstSRS = GEDI.Preprocess_GEDI().get_WKT()
        gedi_path = join(GEDI.Preprocess_GEDI().data_dir,'tif','gedi_2019-2023_clipped.tif')
        outdir_hls = join(self.data_dir,'patches/gedi')
        T.mkdir(outdir_hls,force=True)

        PATCH_SIZE = 64
        STRIDE = 16  # 可改成112表示50%重叠
        random.seed(42)
        with rasterio.open(gedi_path) as src_hls:
            h, w = src_hls.height, src_hls.width
            ds = gdal.Open(gedi_path)
            gt = ds.GetGeoTransform()
            xres = gt[1]
            yres = gt[5]
            x_min = gt[0]
            y_max = gt[3]
            x_max = x_min + ds.RasterXSize * gt[1]
            y_min = y_max + ds.RasterYSize * gt[5]
            count = 0
            samples = []

            for row in tqdm(range(0, h - PATCH_SIZE, STRIDE), desc="Sliding rows"):
            # for row in tqdm(range(0, 1), desc="Sliding rows"):
                for col in range(0, w - PATCH_SIZE, STRIDE):
                    gedi_patch = src_hls.read(
                        window=((row, row + PATCH_SIZE), (col, col + PATCH_SIZE))
                    ).astype(np.float32)
                    # 检查是否全NaN
                    if np.isnan(gedi_patch).all():
                        continue
                    for band in gedi_patch:
                        band[band<-9999] = np.nan

                    patch_name = f"patch_{count:05d}"
                    patch_fpath = join(outdir_hls,patch_name+'.tif')
                    # x_min_all, xres, 0, y_max_all, 0, yres
                    x_min_i = x_min + col * xres
                    y_max_i = y_max + row * yres
                    self.patch_to_tif(patch_fpath, gedi_patch, PATCH_SIZE, x_min_i, xres, y_max_i, yres,dstSRS)
                    count += 1
                    # exit()

        print(f"Total patches saved: {count}")
        return samples


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

class Split_Patch:

    def __init__(self):
        self.data_dir = join(this_script_root,'Split_patch')
        pass

    def run(self):
        self.copy_HLS()
        self.copy_gedi()
        pass

    def gen_random_path_list(self):
        random.seed(42)
        outdir = join(self.data_dir,'patches')
        fpath_dir = join(GenPatch().data_dir,'patches/gedi')
        fpath_list = T.listdir(fpath_dir)
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1
        random.shuffle(fpath_list)
        train_flist = fpath_list[:int(len(fpath_list)*train_ratio)]
        val_flist = fpath_list[int(len(fpath_list)*train_ratio):int(len(fpath_list)*(train_ratio+val_ratio))]
        test_flist = fpath_list[int(len(fpath_list)*(train_ratio+val_ratio)):]

        return train_flist, val_flist, test_flist

    def copy_HLS(self):
        train_flist, val_flist, test_flist = self.gen_random_path_list()
        patch_fdir = join(GenPatch().data_dir,'patches','hls_1km')
        train_fdir = join(self.data_dir,'HLS','train')
        val_fdir = join(self.data_dir,'HLS','val')
        test_fdir = join(self.data_dir,'HLS','test')
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

    def copy_gedi(self):
        train_flist, val_flist, test_flist = self.gen_random_path_list()
        patch_fdir = join(GenPatch().data_dir,'patches','gedi')
        train_fdir = join(self.data_dir,'GEDI','train')
        val_fdir = join(self.data_dir,'GEDI','val')
        test_fdir = join(self.data_dir,'GEDI','test')
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
    GenPatch().run()
    # Split_Patch().run()
    pass

if __name__ == '__main__':

    main()