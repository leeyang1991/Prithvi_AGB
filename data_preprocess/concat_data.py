import numpy as np

from __init__ import *
from __global__ import *
import rasterio

this_script_root = join(data_root,'concat_data')

class Concat_Data:
    def __init__(self):
        self.data_dir = this_script_root
        pass

    def run(self):
        # self.Concat_1km()
        self.Concat_30m()
        pass

    def Concat_1km(self):
        outdir = join(self.data_dir,'tif')
        T.mkdir(outdir,force=True)
        band_list = global_band_list
        import HLS
        import additional_index
        HLS_fpath = join(HLS.Preprocess_HLS().data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7_1km.tif')
        DEM_fpath = join(additional_index.DEM_1km().data_dir,'merge/DEM_1km_reproj6933_crop.tif')
        ndvi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,'ndvi_1km.tif')
        mndwi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,'mndwi_1km.tif')
        nbr_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,'nbr_1km.tif')
        ndwi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,'ndwi_1km.tif')

        HLS_read = rasterio.open(HLS_fpath)
        DEM_read = rasterio.open(DEM_fpath)
        ndvi_read = rasterio.open(ndvi_fpath)
        mndwi_read = rasterio.open(mndwi_fpath)
        nbr_read = rasterio.open(nbr_fpath)
        ndwi_read = rasterio.open(ndwi_fpath)

        HLS_data = HLS_read.read()
        DEM_data = DEM_read.read()
        ndvi_data = ndvi_read.read()
        mndwi_data = mndwi_read.read()
        nbr_data = nbr_read.read()
        ndwi_data = ndwi_read.read()
        stack_data = np.concatenate((HLS_data, DEM_data, ndvi_data, mndwi_data, nbr_data, ndwi_data), axis=0)
        profile = HLS_read.profile
        profile.update(count=stack_data.shape[0])
        # pprint(profile)
        # exit()
        outf = join(outdir,'concat_1km.tif')
        with rasterio.open(outf, "w", **profile) as dst:
            for i in range(stack_data.shape[0]):
                dst.write(stack_data[i], i+1)
                dst.set_band_description(i+1, band_list[i])
        print('done')

    def Concat_30m(self):
        outdir = join(self.data_dir,'tif')
        T.mkdir(outdir,force=True)
        band_list = global_band_list
        import HLS
        import additional_index
        HLS_fpath = join(HLS.Preprocess_HLS().data_dir,'reproj_qa_concatenate_aggragate_tif_mosaic_merge-bands/B2-B7.tif')
        DEM_fpath = join(additional_index.DEM_30m().data_dir,'merge/DEM_30m_reproj6933_crop.tif')
        ndvi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,'ndvi_30m.tif')
        mndwi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,'mndwi_30m.tif')
        nbr_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,'nbr_30m.tif')
        ndwi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,'ndwi_30m.tif')

        HLS_read = rasterio.open(HLS_fpath)
        DEM_read = rasterio.open(DEM_fpath)
        ndvi_read = rasterio.open(ndvi_fpath)
        mndwi_read = rasterio.open(mndwi_fpath)
        nbr_read = rasterio.open(nbr_fpath)
        ndwi_read = rasterio.open(ndwi_fpath)

        HLS_data = HLS_read.read()
        DEM_data = DEM_read.read()
        ndvi_data = ndvi_read.read()
        mndwi_data = mndwi_read.read()
        nbr_data = nbr_read.read()
        ndwi_data = ndwi_read.read()
        stack_data = np.concatenate((HLS_data, DEM_data, ndvi_data, mndwi_data, nbr_data, ndwi_data), axis=0)
        profile = HLS_read.profile
        profile.update(count=stack_data.shape[0])
        # pprint(profile)
        # exit()
        outf = join(outdir,'concat_30m.tif')
        with rasterio.open(outf, "w", **profile) as dst:
            for i in range(stack_data.shape[0]):
                dst.write(stack_data[i], i+1)
                dst.set_band_description(i+1, band_list[i])
        print('done')

def main():
    Concat_Data().run()
    pass

if __name__ == '__main__':
    main()