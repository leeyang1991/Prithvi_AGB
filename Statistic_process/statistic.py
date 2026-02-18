import matplotlib.pyplot as plt

from utils import *

T = Tools()

this_script_root = join(results_root,'statistic')

class Temporal_statistic:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = (
            T.mk_class_dir('Temporal_statistic',this_script_root,mode=2))

        pass

    def run(self):
        # self.clip_ndvi()
        # self.mask_invalid_agb_values()
        self.temporal()

        pass

    def temporal(self):
        fdir = join(self.this_class_tif,'mask_invalid_agb_values')
        outdir = join(self.this_class_png, 'temporal')
        T.mkdir(outdir,force=True)

        year_list = []
        value_list = []
        std_list = []

        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            year = f.split('_')[0]
            year_list.append(year)

            fpath = join(fdir, f)
            data,profile = RasterIO_Func().read_tif(fpath)
            print(data.shape)
            # plt.imshow(data,vmin=0,vmax=150)
            # plt.colorbar()
            # plt.show()
            # pause()
            mean_agb = np.nanmean(data)
            std_agb = np.nanstd(data)
            print(mean_agb)
            value_list.append(mean_agb)
            std_list.append(std_agb)
        # plt.bar(year_list,value_list,yerr=std_list, capsize=5)
        plt.bar(year_list,value_list)
        plt.xlabel('Year')
        plt.ylabel('Mean AGB')
        plt.title('Temporal Variation of Mean AGB')
        plt.ylim(15.3,15.5)
        plt.savefig(join(outdir,'temporal_variation_agb.pdf'))
        plt.savefig(join(outdir,'temporal_variation_agb.png'))
        plt.close()
        pass

    def clip_ndvi(self):
        ndvi_fpath = join(data_root, 'additional_index/HLS_Vegetation_Index_annual/30m/AZ/2019/ndvi.tif')
        agb_fpath = join(this_root,'train/predict_agb_annual/AZ/mosaic/2019_agb_30m_8_bands_1792.tif')
        out_raster = join(self.this_class_arr,'ndvi_clip_agb.tif')
        # print(bounds)
        # exit()
        RasterIO_Func_Extend().clip_tif_by_tif(ndvi_fpath,out_raster,agb_fpath)

    def mask_invalid_agb_values(self):
        agb_fdir = join(this_root, 'train/predict_agb_annual/AZ/mosaic')

        ndvi_fpath = join(self.this_class_arr,'ndvi_clip_agb.tif')
        outdir = join(self.this_class_tif,'mask_invalid_agb_values')
        T.mkdir(outdir,force=True)
        outf = join(outdir,'ndvi_mask.npy')
        if isfile(outf):
            mask = np.load(outf)
        else:
            data, profile = RasterIO_Func().read_tif(ndvi_fpath)
            mask1 = data > 0
            mask2 = data < 1
            mask = mask1 & mask2
            np.save(outf,mask)

        for f in T.listdir(agb_fdir):
            print(f)
            fpath = join(agb_fdir, f)
            data, profile = RasterIO_Func().read_tif(fpath)
            data[~mask] = np.nan
            RasterIO_Func().write_tif(data, join(outdir,f), profile)


def main():
    Temporal_statistic().run()

    pass


if __name__ == '__main__':
    main()