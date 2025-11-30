import numpy as np

from __init__ import *
from __global__ import *
import rasterio

this_script_root = join(data_root,'concat_data')

class Concat_Data:
    def __init__(self):
        self.data_dir = this_script_root
        self.resolution = '30m'
        # self.resolution = '1km'

        self.region = 'AZ'
        # self.region = 'NM'
        pass

    def run(self):
        self.Concat()
        # self.build_pyramid()
        pass


    def Concat(self):
        outdir = join(self.data_dir,f'{self.resolution}')
        T.mkdir(outdir,force=True)
        # band_list = global_band_list
        band_list = global_band_list_8
        import HLS
        import additional_index
        if self.resolution == '30m':
            HLS_fpath = join(HLS.Preprocess_HLS().data_dir,f'1.4_mosaic/{self.region}_6_bands.tif')
            DEM_fpath = join(additional_index.DEM_30m(self.region).data_dir,f'merge/DEM_30m_reproj6933_crop.tif')
            ndvi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,self.resolution,self.region,'ndvi.tif')
            # mndwi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,self.resolution,self.region,'mndwi.tif')
            # nbr_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,self.resolution,self.region,'nbr.tif')
            # ndwi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir,self.resolution,self.region,'ndwi.tif')
            outf = join(outdir,f'{self.region}_concat_30m.tif')

        elif self.resolution == '1km':
            HLS_fpath = join(HLS.Preprocess_HLS().data_dir, f'1.5_resample_30m_to_1km/{self.region}_6_bands_resample_1km.tif')
            DEM_fpath = join(additional_index.DEM_1km().data_dir, f'{self.region}/DEM_1km_{self.region}.tif')
            ndvi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir, self.resolution, self.region,
                              'ndvi.tif')
            # mndwi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir, self.resolution, self.region,
            #                    'mndwi.tif')
            # nbr_fpath = join(additional_index.HLS_Vegetation_Index().data_dir, self.resolution, self.region, 'nbr.tif')
            # ndwi_fpath = join(additional_index.HLS_Vegetation_Index().data_dir, self.resolution, self.region,
            #                   'ndwi.tif')
            outf = join(outdir,f'{self.region}_concat_1km.tif')

        else:
            raise
        print('reading data...')
        HLS_data = rasterio.open(HLS_fpath).read()
        DEM_data = rasterio.open(DEM_fpath).read()
        ndvi_data = rasterio.open(ndvi_fpath).read()
        # mndwi_data = rasterio.open(mndwi_fpath).read()
        # nbr_data = rasterio.open(nbr_fpath).read()
        # ndwi_data = rasterio.open(ndwi_fpath).read()
        # stack_data = np.concatenate((HLS_data, DEM_data, ndvi_data, mndwi_data, nbr_data, ndwi_data), axis=0)
        stack_data = np.concatenate((HLS_data, DEM_data, ndvi_data), axis=0)

        print('reading done')

        profile = rasterio.open(HLS_fpath).profile

        profile.update(count=stack_data.shape[0],dtype=np.float32)
        if self.resolution == '30m':
            profile.update(
                bigtiff='YES',
            )
        # pprint(profile)
        # exit()
        print('writing data...')
        with rasterio.open(outf, "w", **profile) as dst:
            for i in tqdm(range(stack_data.shape[0]),desc='writing bands'):
                dst.write(stack_data[i], i+1)
                dst.set_band_description(i+1, band_list[i])
        print('done')


    def build_pyramid(self):
        import HLS
        fdir = join(self.data_dir, f'{self.resolution}')
        fpath = join(fdir, f'{self.region}_concat_{self.resolution}.tif')
        # 'concat_1km.tif'
        print('building pyramid...')
        HLS.RasterIO_Func().build_pyramid(fpath,bigtiff='YES')
        print('done')

def main():
    Concat_Data().run()
    pass

if __name__ == '__main__':
    main()