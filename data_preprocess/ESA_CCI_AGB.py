from utils import *
import xarray as xr

from rasterio.transform import Affine
from rasterio.crs import CRS
T = Tools()


class ESA_CCI_AGB:

    def __init__(self):
        self.datadir = join(data_root,'ESA_CCI_AGB')

        pass

    def run(self):
        # self.nc_to_tif()
        self.cal_mean()
        pass


    def nc_to_tif(self):
        fdir = '/home/yangli/HDD14T/AGB_raw/ESACCI_AGB_global'
        outdir = join(self.datadir,'tif')
        T.mkdir(outdir,force=True)
        profile_template = self.profile_template()
        for f in T.listdir(fdir):
            if not f.endswith('.nc'):
                continue
            outf = join(outdir,f.replace('.nc','.tif'))
            fpath = join(fdir,f)
            ds = xr.open_dataset(
                fpath,
                chunks={'lat': 1000, 'lon': 1000}
            )
            print(ds)
            # exit()
            subset = ds.sel(
                lon=slice(-115, -108),
                # lat=slice(30, 37)
                lat=slice(37, 30)
            )
            data = subset['agb']
            data = data.squeeze()

            # print(data.shape)

            profile_template['height'] = data.shape[0]
            profile_template['width'] = data.shape[1]
            profile_template['dtype'] = np.float32
            Affine_ = Affine(0.00088889, 0.0, -115.0,
                             0.0, -0.00088889, 37.0)
            profile_template['transform'] = Affine_
            # pprint(profile_template)
            RasterIO_Func().write_tif(data,outf, profile_template)
            print('done')
            print(outf)
            # exit()



        pass

    def profile_template(self):
        profile = {'blockxsize': 432,
                 'blockysize': 224,
                 'compress': 'packbits',
                 'count': 1,
                 'crs': CRS().from_wkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
                                     'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
                                     'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
                                     'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'),
                 'driver': 'GTiff',
                 'dtype': 'uint16',
                 'height': 2160,
                 'interleave': 'pixel',
                 'nodata': None,
                 'tiled': True,
                 'transform': Affine(0.08333333333333333, 0.0, -180.0,
                       0.0, -0.08333333333333333, 90.0),
                 'width': 4320}

        return profile

    def cal_mean(self):
        fdir = join(self.datadir,'tif')
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            data, profile = RasterIO_Func().read_tif(fpath)
            mean_agb = np.nanmean(data)
            print(f,mean_agb)

        pass

def main():
    ESA_CCI_AGB().run()

if __name__ == '__main__':
    main()