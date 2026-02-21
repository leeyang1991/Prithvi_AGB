
from utils import *

T = Tools()


class PDSI:

    def __init__(self):
        self.datadir = join(data_root,'PDSI')
        pass


    def run(self):
        # self.reproj()
        # self.clip_tif()
        # self.time_series()
        self.aggragate_monthly_to_annual()
        pass

    def reproj(self):
        fdir = join(self.datadir, 'tif')
        outdir = join(self.datadir, 'tif_reproj')
        T.mkdir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            outf = join(outdir,f)
            RasterIO_Func_Extend().reproject_tif(fpath,outf, dst_crs='EPSG:6933')

        pass

    def clip_tif(self):
        fdir = join(self.datadir,'tif_reproj')
        outdir = join(self.datadir,'tif_clip')
        T.mkdir(outdir,force=True)
        target_tif_path = join(this_root,'train/predict_agb_annual/AZ/mosaic','2022_agb_30m_8_bands_1792.tif')
        bounds = RasterIO_Func_Extend().get_tif_bounds_(target_tif_path)
        # print(bounds)
        # exit()
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            RasterIO_Func_Extend().clip_tif_by_bounds(fpath,outf,bounds)

        pass

    def time_series(self):
        fdir = join(self.datadir,'tif_clip')
        outdir = join(self.datadir,'time_series')
        T.mkdir(outdir,force=True)
        date_list = []
        mean_list = []
        for f in tqdm(T.listdir(fdir)):
            year = f.split('.')[0][:4]
            month = f.split('.')[0][4:6]
            date_obj = datetime.datetime(int(year),int(month),1)
            fpath = join(fdir,f)
            data,_ = RasterIO_Func().read_tif(fpath)
            data[data<-9999] = np.nan
            mean = np.nanmean(data)
            date_list.append(date_obj)
            mean_list.append(mean)
        plt.plot(date_list,mean_list)
        plt.xlabel('Date')
        plt.ylabel('Mean PDSI')
        plt.savefig(join(outdir,'pdsi_time_series.pdf'))
        plt.savefig(join(outdir,'pdsi_time_series.png'))
        plt.close()
        print(outdir)

        pass

    def aggragate_monthly_to_annual(self):
        fdir = join(self.datadir,'tif_clip')
        outdir = join(self.datadir,'tif_clip_annual')
        T.mkdir(outdir,force=True)
        data_dict = {}
        for f in tqdm(T.listdir(fdir)):
            year = f.split('.')[0][:4]
            month = f.split('.')[0][4:6]
            date_obj = datetime.datetime(int(year),int(month),1)
            fpath = join(fdir,f)
            data,profile = RasterIO_Func().read_tif(fpath)
            data[data<-9999] = np.nan
            if not year in data_dict:
                data_dict[year] = []
            data_dict[year].append(data)
        for year in tqdm(data_dict):
            data_list = data_dict[year]
            data_stack = np.stack(data_list,axis=0)
            data_mean = np.nanmean(data_stack,axis=0)
            outf = join(outdir,f'{year}.tif')
            RasterIO_Func().write_tif(data_mean,outf,profile)

def main():
    PDSI().run()
    pass

if __name__ == "__main__":

    main()