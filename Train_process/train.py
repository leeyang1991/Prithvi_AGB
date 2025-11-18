from __global__ import *
import os
print(os.getcwd())
# exit()
import sys
from lytools import *
import numpy as np
import torch
import gdown
import rasterio
import terratorch
import albumentations
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from pathlib import Path
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
import warnings
import os
import zipfile
from lightning.pytorch.callbacks import Callback
import torch
from segmentation_models_pytorch.base.initialization import initialize_decoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch import nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
warnings.filterwarnings('ignore')

T = Tools()


def init_datamodule_train():
    dataset_path = Path(join(data_root,'Patch','Split_patch'))
    print(dataset_path)
    means = [
        547.36707,
        898.5121,
        1020.9082,
        2665.5352,
        2340.584,
        1610.1407,

        15000000,
        -1000,
        -1000,
        -1000,
        -1000,
    ]
    means = np.array(means) / 10000.
    means_list = means.tolist()

    stds = [
        411.4701,
        558.54065,
        815.94025,
        812.4403,
        1113.7145,
        1067.641,

        2000000,
        -100,
        -100,
        -100,
        -100,
    ]
    stds = np.array(stds) / 10000.
    stds_list = stds.tolist()

    datamodule = terratorch.datamodules.GenericNonGeoPixelwiseRegressionDataModule(
        batch_size=20,
        num_workers=8,
        # num_classes=6,
        check_stackability=False,
        # Define dataset paths
        train_data_root=dataset_path / 'concat_1km/train/',
        train_label_data_root=dataset_path / 'GEDI/train/',
        val_data_root=dataset_path / 'concat_1km/val/',
        val_label_data_root=dataset_path / 'GEDI/val/',
        test_data_root=dataset_path / 'concat_1km/test',
        test_label_data_root=dataset_path / 'GEDI/test',

        img_grep='*.tif',
        label_grep='*.tif',

        train_transform=[
            albumentations.D4(),  # Random flips and rotation
            albumentations.pytorch.transforms.ToTensorV2(),
        ],
        val_transform=None,  # Using ToTensor() by default
        test_transform=None,

        # Define standardization values
        means=means_list,
        stds=stds_list,
        dataset_bands=global_band_list,
        output_bands=global_band_list,
        rgb_indices=[2, 1, 0],
        no_data_replace=0,
        no_label_replace=-1,
    )
    return datamodule

def init_datamodule_predict_30m():
    dataset_path = Path(join(data_root,'Patch','patches'))
    print(dataset_path)
    means = [
        547.36707,
        898.5121,
        1020.9082,
        2665.5352,
        2340.584,
        1610.1407,

        15000000,
        -1000,
        -1000,
        -1000,
        -1000,
    ]
    means = np.array(means) / 10000.
    means_list = means.tolist()

    stds = [
        411.4701,
        558.54065,
        815.94025,
        812.4403,
        1113.7145,
        1067.641,

        2000000,
        -100,
        -100,
        -100,
        -100,
    ]
    stds = np.array(stds) / 10000.
    stds_list = stds.tolist()

    datamodule = terratorch.datamodules.GenericNonGeoPixelwiseRegressionDataModule(
        batch_size=50,
        num_workers=8,
        # num_classes=6,
        check_stackability=False,
        # Define dataset paths
        train_data_root=dataset_path / 'concat_30m/',
        train_label_data_root=dataset_path / 'concat_30m/',
        val_data_root=dataset_path / 'concat_30m/',
        val_label_data_root=dataset_path / 'concat_30m/',
        test_data_root=dataset_path / 'concat_30m/',
        test_label_data_root=dataset_path / 'concat_30m/',

        img_grep='*.tif',
        label_grep='*.tif',

        train_transform=[
            albumentations.D4(),  # Random flips and rotation
            albumentations.pytorch.transforms.ToTensorV2(),
        ],
        val_transform=None,  # Using ToTensor() by default
        test_transform=None,

        # Define standardization values
        means=means_list,
        stds=stds_list,
        dataset_bands=global_band_list,
        output_bands=global_band_list,
        rgb_indices=[2, 1, 0],
        no_data_replace=0,
        no_label_replace=-1,
    )
    return datamodule

@TERRATORCH_DECODER_REGISTRY.register
class UNetDecoder1(nn.Module):
    """UNetDecoder. Wrapper around UNetDecoder from segmentation_models_pytorch to avoid ignoring the first layer."""

    def __init__(
        self, embed_dim: list[int], channels: list[int], use_batchnorm: bool = True, attention_type: str | None = None):
        """Constructor

        Args:
            embed_dim (list[int]): Input embedding dimension for each input.
            channels (list[int]): Channels used in the decoder.
            use_batchnorm (bool, optional): Whether to use batchnorm. Defaults to True.
            attention_type (str | None, optional): Attention type to use. Defaults to None
        """
        super().__init__()
        self.decoder = UnetDecoder(
            encoder_channels=[embed_dim[0], *embed_dim],
            decoder_channels=channels,
            n_blocks=len(channels),
            use_norm=use_batchnorm,
            add_center_block=False,
            attention_type=attention_type,
        )
        initialize_decoder(self.decoder)
        self.out_channels = channels[-1]

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        # The first layer is ignored in the original UnetDecoder, so we need to duplicate the first layer
        x = [x[0].clone(), *x]
        # print(len(x))
        # exit()
        return self.decoder(x)


def init_model():
    model = terratorch.tasks.PixelwiseRegressionTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            # Backbone
            "backbone": "prithvi_eo_v2_300",
            # Model can be either prithvi_eo_v1_100, prithvi_eo_v2_300, prithvi_eo_v2_300_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
            "backbone_pretrained": True,
            "backbone_num_frames": 1,  # 1 is the default value,
            # "backbone_img_size": 224,
            # "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "backbone_bands": global_band_list,
            # "backbone_coords_encoding": [], # use ["time", "location"] for time and location metadata

            # Necks
            "necks": [
                {
                    "name": "SelectIndices",
                    # "indices": [2, 5, 8, 11] # indices for prithvi_eo_v1_100
                    "indices": [5, 11, 17, 23]  # indices for prithvi_eo_v2_300
                    # "indices": [7, 15, 23, 31] # indices for prithvi_eo_v2_600
                },
                {"name": "ReshapeTokensToImage", },
                {"name": "LearnedInterpolateToPyramidal"}
            ],

            # Decoder
            "decoder": "UNetDecoder1",
            # "decoder_channels": [512, 256, 128, 64],
            "decoder_channels": [256, 128, 64, 32],
            # "head_dropout": 0.16194593880230534,
            # "head_final_act": torch.nn.ReLU,
            # "head_learned_upscale_layers": 2
        },

        loss="rmse",
        optimizer="AdamW",
        lr=1e-3,
        ignore_index=-1,
        freeze_backbone=True,  # Only to speed up fine-tuning
        freeze_decoder=False,
        plot_on_val=False,
        # class_names=['no burned', 'burned']  # optionally define class names
    )

    return model

def init_trainer():
    trainer_dir = join(this_root,'model','trainer')
    T.mkdir(trainer_dir,force=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # dirpath="output/agb/checkpoints/",
        dirpath=join(trainer_dir,'checkpoints'),
        mode="min",
        monitor="val/RMSE",  # Variable to monitor
        filename="best-{epoch:02d}",
    )

    trainer = pl.Trainer(
        accelerator=global_device,
        strategy="auto",
        devices=1,  # Deactivate multi-gpu because it often fails in notebooks
        precision='bf16-mixed',  # Speed up training
        num_nodes=1,
        logger=True,  # Uses TensorBoard by default
        max_epochs=100,  # For demos
        log_every_n_steps=1,
        enable_checkpointing=True,
        # callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
        callbacks=[checkpoint_callback, pl.callbacks.TQDMProgressBar()],
        # default_root_dir="output/agb",
        default_root_dir=trainer_dir,
        detect_anomaly=True,
    )
    return trainer

def train_agb():
    trainer = init_trainer()
    model = init_model()
    datamodule = init_datamodule_train()
    # exit()
    trainer.fit(model, datamodule=datamodule)

    pass

def check_performance():
    ckpt_path = join(this_root,'model','trainer/checkpoints/best-epoch=99.ckpt')
    trainer = init_trainer()
    model = init_model()
    datamodule = init_datamodule_train()
    trainer.test(model, datamodule=datamodule,ckpt_path=ckpt_path)
    plt.show()
    pass

def predict_agb(ckpt_path):
    outdir = join(results_root,'agb_pred','patch_30m_11_bands')
    T.mkdir(outdir,force=True)
    model_init = init_model()
    datamodule = init_datamodule_predict_30m()
    datamodule.setup("test")
    test_dataset = datamodule.test_dataset
    # print(len(test_dataset));exit()
    # let's run the model on the test set
    # trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # now we can use the model for predictions and plotting!
    model = terratorch.tasks.PixelwiseRegressionTask.load_from_checkpoint(
        ckpt_path,
        model_factory=model_init.hparams.model_factory,
        model_args=model_init.hparams.model_args,
    )

    test_loader = datamodule.test_dataloader()
    with torch.no_grad():
        # batch = next(iter(test_loader))
        for batch in tqdm(test_loader):
            batch = datamodule.aug(batch)
            images = batch["image"].to(model.device)

            masks = batch["mask"].numpy()
            filename_list = batch['filename']
            preds = model(images).output
            # print(batch.keys())

            for i in range(len(preds)):
                preds_image = preds[i].cpu().numpy()
                patch_filename = filename_list[i]
                save_pred_image(preds_image, patch_filename,outdir)
            # exit()

@Decorator.shutup_gdal
def save_pred_image(preds_image, patch_filename,outdir):
    dstSRS = global_gedi_WKT()
    pred_fpath = join(outdir,patch_filename.split('/')[-1])
    # exit()
    ds = gdal.Open(patch_filename)
    gt = ds.GetGeoTransform()
    xres = gt[1]
    yres = gt[5]
    x_min = gt[0]
    y_max = gt[3]
    x_max = x_min + ds.RasterXSize * gt[1]
    y_min = y_max + ds.RasterYSize * gt[5]
    driver = gdal.GetDriverByName('GTiff')
    PATCH_SIZE = np.shape(preds_image)[0]
    out_ds = driver.Create(pred_fpath, PATCH_SIZE, PATCH_SIZE, 1, gdal.GDT_Float32,
                           options=['COMPRESS=LZW', 'BIGTIFF=YES'])
    out_gt = (x_min, xres, 0, y_max, 0, yres)
    out_ds.SetGeoTransform(out_gt)
    out_ds.SetProjection(dstSRS)
        # band = hls_patch[idx-1]
    out_ds.GetRasterBand(1).WriteArray(preds_image)
    out_ds.GetRasterBand(1).SetNoDataValue(global_nodata_value)
    # out_band.FlushCache()

    out_ds = None
    pass


@Decorator.shutup_gdal
def mosaic_spatial_tifs_no_overlap():
    nodata_value = global_nodata_value
    dstSRS = global_gedi_WKT()
    fdir = '/home/yangli/Projects_Data/terratorch_learn/pred'
    outdir = '/home/yangli/Projects_Data/terratorch_learn/pred_mosaic'
    outf = join(outdir,f'mosaic5.tif')

    T.mkdir(outdir)
    fpath_list = []
    for f in T.listdir(fdir):
        fpath_list.append(join(fdir,f))
    # print(fpath_list)
    # exit()

    ref_ds = gdal.Open(fpath_list[0])
    gt = ref_ds.GetGeoTransform()
    xres = gt[1]
    yres = gt[5]
    tiles_info = []

    for path in fpath_list:
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

    for path, xmin, xmax, ymin, ymax in tqdm(tiles_info):
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
    out_ds = driver.Create(outf, cols, rows, 1, gdal.GDT_Float32,
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
def mosaic_spatial_tifs_overlap():
    patch_size = 224
    stride = 112
    nodata_value = global_nodata_value
    dstSRS = global_gedi_WKT()
    # fdir = join(results_root,'agb_pred','patch_30m')
    fdir = join(results_root,'agb_pred','patch_30m_11_bands')
    outdir = join(results_root,'agb_pred','mosaic_11_bands')
    outf = join(outdir,f'agb_30m_11_bands.tif')

    T.mkdir(outdir)
    fpath_list = []
    for f in T.listdir(fdir):
        fpath_list.append(join(fdir,f))
    # print(fpath_list)
    # exit()

    ref_ds = gdal.Open(fpath_list[0])
    gt = ref_ds.GetGeoTransform()
    xres = gt[1]
    yres = gt[5]
    tiles_info = []

    for path in fpath_list:
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
    # print(cols, rows)
    # exit()
    AGB_sum = np.ones((rows, cols), dtype=np.float32)
    AGB_weight = np.zeros((rows, cols), dtype=np.float32)
    # nodata = nodata_value

    w = np.hanning(patch_size)
    weight2d = np.outer(w, w)
    weight2d /= weight2d.max()


    for path, xmin, xmax, ymin, ymax in tqdm(tiles_info):
        ds = gdal.Open(path)
        data = ds.ReadAsArray().astype(np.float32)
        # data[data<-9999] = 0
        gt_i = ds.GetGeoTransform()

        # tile 在 mosaic 中的起始、结束行列号
        x_off = int((xmin - x_min_all) / xres)
        y_off = int((y_max_all - ymax) / abs(yres))
        h, w = ds.RasterYSize, ds.RasterXSize
        data[np.isnan(data)] = 0
        data = data * weight2d

        AGB_sum[y_off:y_off + h, x_off:x_off + w] += data
        AGB_weight[y_off:y_off + h, x_off:x_off + w] += weight2d

    AGB_map = AGB_sum / np.maximum(AGB_weight, 1e-6)
    AGB_map = AGB_map[stride:-stride, stride:-stride]
    # print(np.shape(AGB_map))
    # print(cols, rows)
    # exit()
    # plt.imshow(AGB_map)
    # plt.show()
    # pause()
    driver = gdal.GetDriverByName('GTiff')
    cols_new = cols
    out_ds = driver.Create(outf, AGB_map.shape[1], AGB_map.shape[0], 1, gdal.GDT_Float32,
                           options=['COMPRESS=LZW', 'BIGTIFF=YES'])
    # out_gt = (x_min_all, xres, 0, y_max_all, 0, yres)
    out_gt = (x_min_all + stride * xres, xres, 0, y_max_all + stride * yres, 0, yres)
    out_ds.SetGeoTransform(out_gt)
    out_ds.SetProjection(dstSRS)

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(AGB_map)
    out_band.SetNoDataValue(nodata_value)
    out_band.FlushCache()

    out_ds = None
    pass

def resample_30_to_1km():
    fpath = join(results_root, 'agb_pred', 'mosaic', f'agb_30m.tif')
    outpath = join(results_root, 'agb_pred', 'mosaic', f'agb_30m_resample_1km.tif')
    res = global_res_gedi
    SRS = global_gedi_WKT()
    ToRaster().resample_reproj(fpath, outpath, res, srcSRS=SRS, dstSRS=SRS)
    pass

def benchmark():
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import from_bounds
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    pred_agb_fpath = join(results_root, 'agb_pred', 'mosaic', f'agb_30m_resample_1km.tif')
    obs_agb_fpath = join(data_root,'GEDI/tif/gedi_2019-2023_clipped.tif')

    src_obs = rasterio.open(obs_agb_fpath)
    obs = src_obs.read(1)
    obs_profile = src_obs.profile
    obs_bounds = src_obs.bounds
    obs_transform = src_obs.transform
    obs_crs = src_obs.crs

    src_pred = rasterio.open(pred_agb_fpath)
    pred = src_pred.read(1)
    pred_bounds = src_pred.bounds
    pred_transform = src_pred.transform
    pred_crs = src_pred.crs
    plt.imshow(obs,vmin=0,vmax=150,cmap='jet')
    plt.figure()
    plt.imshow(pred,vmin=0,vmax=150,cmap='jet')
    plt.show()
    # exit()
    xmin = max(obs_bounds.left, pred_bounds.left)
    xmax = min(obs_bounds.right, pred_bounds.right)
    ymin = max(obs_bounds.bottom, pred_bounds.bottom)
    ymax = min(obs_bounds.top, pred_bounds.top)
    print(xmin, xmax, ymin, ymax)
    window_obs = from_bounds(xmin, ymin, xmax, ymax, obs_transform)
    obs_crop = src_obs.read(1, window=window_obs)
    transform_obs_crop = src_obs.window_transform(window_obs)
    # plt.imshow(obs_crop,vmin=0,vmax=100,cmap='jet')
    # plt.figure()
    # plt.imshow(pred,vmin=0,vmax=100,cmap='jet')
    # plt.show()
    # print(np.shape(obs_crop))
    # print(np.shape(pred))
    # lim=150

    # obs_crop[obs_crop>lim] = np.nan
    # pred[pred>lim] = np.nan
    obs_crop_flatten = obs_crop.flatten()
    pred_flatten = pred.flatten()
    mask = ~np.isnan(obs_crop_flatten) & ~np.isnan(pred_flatten)
    obs_crop_flatten = obs_crop_flatten[mask]
    pred_flatten = pred_flatten[mask]
    print(len(obs_crop_flatten))
    print(len(pred_flatten))
    r2 = r2_score(obs_crop_flatten, pred_flatten)
    print(r2)

    KDE_plot().plot_scatter(obs_crop_flatten, pred_flatten,is_plot_1_1_line=True,is_equal=True)
    # val_range_index = random.sample(list(range(len(pred_flatten))), int(len(pred_flatten)/10))
    # obs_crop_flatten = obs_crop_flatten[val_range_index]
    # pred_flatten = pred_flatten[val_range_index]
    # plt.scatter(obs_crop_flatten, pred_flatten,s=10,alpha=0.5,c='gray',edgecolors="none",marker='.')
    # lim = 150
    # plt.plot([0, lim], [0, lim], 'r--')
    # plt.xlim(0, lim)
    # plt.ylim(0, lim)
    # plt.axis('equal')
    plt.xlabel('Observed AGB')
    plt.ylabel('Predicted AGB')
    plt.title(f'R2 = {r2:.3f}')
    plt.show()

def main():
    # best_ckpt_path = '/home/yangli/Remote_SSH_Pyproject/terratorch_learn/Train_process/output/agb/checkpoints/best-epoch=86.ckpt'
    # print(isfile(best_ckpt_path))
    # train_agb()
    # check_performance()
    # best_ckpt_path = join(this_root,'model','trainer/checkpoints/best-epoch=99.ckpt')
    # predict_agb(best_ckpt_path)
    mosaic_spatial_tifs_overlap()
    # resample_30_to_1km()
    # benchmark()

    pass


if __name__ == '__main__':
    main()
