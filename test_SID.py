from options.eld.base_options import BaseOptions 
from engine import Engine
import torch
import torch.backends.cudnn as cudnn
import dataset.sid_dataset as datasets
import dataset
import noise

opt = BaseOptions().parse()

cudnn.benchmark = True

datadir = './datasets/SID/Sony'

expo_ratio = [100, 250, 300] # [100, 250, 300] # , 100, 250
read_expo_ratio = lambda x: float(x.split('_')[-1][:-5])

train_fns = dataset.read_paired_fns('./dataset/Sony_train.txt')
eval_fns = dataset.read_paired_fns('./dataset/Sony_val.txt')
test_fns = dataset.read_paired_fns('./dataset/Sony_test.txt')

eval_fns_list = [[fn for fn in eval_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]
test_fns_list = [[fn for fn in test_fns if min(int(read_expo_ratio(fn[1])/read_expo_ratio(fn[0])), 300)==ratio] for ratio in expo_ratio]
eval_fns_list = [lst_1 + lst_2 for lst_1, lst_2 in zip(eval_fns_list, test_fns_list)]

# evaluate 15 indoor scenes (but you can also evaluate the performance on the whole dataset)
indoor_ids = dataset.read_paired_fns('./SID_Sony_15_paired.txt')
# indoor_ids = dataset.read_paired_fns('./SID_ablation.txt')
# indoor_ids = dataset.read_paired_fns('./SID_Sony_paired.txt')
eval_fns_list = [[(fn[0], fn[1]) for fn in indoor_ids if int(fn[2]) == ratio] for ratio in expo_ratio]

noise_model = noise.NoiseModel(model="P+g", include=4)
eval_datasets = [datasets.SIDDataset(datadir, fns, noise_model, memorize=False, size=None, augment=False, stage_in=opt.stage_in, stage_out=opt.stage_out) for fns in eval_fns_list]


eval_dataloaders = [torch.utils.data.DataLoader(
    eval_dataset, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True) for eval_dataset in eval_datasets]


"""Main Loop"""
engine = Engine(opt)

for ratio, dataloader in zip(expo_ratio, eval_dataloaders):
    # if ratio not in [300]: continue
    print('Eval ratio {}'.format(ratio))

    # Notice for SID dataset, we only evaluate the quantitative results based on the center 512x512 image regions (Enabled by 'crop=True')
    # since we observe some fixed pattern noise appeared at the bottom corner of some pictures.
    # We don't find such noise pattern in our ELD dataset (SonyA7S2 set), which uses the same camera model as SID dataset.
    # As a result, we conjecture this fixed noise pattern might be caused by lens damage, which should be excluded in evaluation.
    
    # PSNR_list = []
    # SSIM_list = []
    # for iter_num in [1, 2,3,5,10]:
    #     print("Iter num:", iter_num)
    #     res = engine.eval(dataloader, dataset_name='sid_eval_{}'.format(ratio), correct=True, crop=True, iter_num=iter_num-1) # additional iter num
    #     PSNR_list.append(res["PSNR"])
    #     SSIM_list.append(res["SSIM"])
    # print(PSNR_list)
    # print(SSIM_list)
    
    #engine.eval(dataloader, dataset_name='sid_eval_{}'.format(ratio), correct=True, crop=True, iter_num=0, savedir=f"images/{opt.model_path.split('/')[-2]}/{ratio}")
    # engine.eval(dataloader, dataset_name='sid_eval_{}'.format(ratio), correct=True, crop=True, iter_num=1, savedir=f"images/{opt.model_path.split('/')[-2]}/{ratio}")
    engine.eval(dataloader, dataset_name='sid_eval_{}'.format(ratio), correct=True, crop=True, iter_num=opt.iter_num, savedir=f"images/{opt.model_path.split('/')[-2]}/{ratio}") 
