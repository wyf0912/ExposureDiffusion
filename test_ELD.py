from options.eld.base_options import BaseOptions
from engine import Engine
import torch
import torch.backends.cudnn as cudnn
import dataset.sid_dataset as datasets
import noise

opt = BaseOptions().parse()

cudnn.benchmark = True

"""Main Loop"""
engine = Engine(opt)

databasedir = '/home/Dataset/DatasetYufei/ELD/'
method = opt.name
scenes = list(range(1, 10+1))
cameras = ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']     
suffixes = ['.CR2', '.CR2', '.CR2', '.nef', '.ARW']


if opt.include is not None:
    cameras = cameras[opt.include:opt.include+1]
    suffixes = suffixes[opt.include:opt.include+1]
else:
    cameras = ['CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']     
    suffixes = ['.CR2', '.CR2', '.nef', '.ARW']


# img_ids_set = [[4, 9, 14]]
img_ids_set = [[4, 9, 14], [5, 10, 15]]
# for scene in scenes:
ratio = [100, 200] # [100, 200]
for i, img_ids in enumerate(img_ids_set):
    # if ratio[i]==100:
    #     continue
    print(img_ids)
    # eval_datasets = [datasets.ELDEvalDataset(databasedir, camera_suffix, scenes=[scene], img_ids=img_ids) for camera_suffix in zip(cameras, suffixes)]
    noise_model = noise.NoiseModel(model="P+g", include=opt.include)
    eval_datasets = [datasets.ELDEvalDataset(databasedir, camera_suffix, noise_model,scenes=scenes, img_ids=img_ids) for camera_suffix in zip(cameras, suffixes)]

    eval_dataloaders = [torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True) for eval_dataset in eval_datasets]

    # cameras = ['CanonEOS5D4', 'HuaweiHonor10']
    psnrs = []
    ssims = []
    for camera, dataloader in zip(cameras, eval_dataloaders):
        print('Eval camera {}'.format(camera))
        
        # we evaluate PSNR/SSIM on full size images
        crop = False
        savedir = f"images/{opt.model_path.split('/')[-2]}/{ratio[i]}"
        res = engine.eval(dataloader, dataset_name='eld_eval_{}'.format(camera), correct=True, iter_num=opt.iter_num, crop=crop, savedir=savedir) 
        
        psnrs.append(res['PSNR'])
        ssims.append(res['SSIM'])
